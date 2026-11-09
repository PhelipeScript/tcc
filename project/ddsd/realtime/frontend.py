"""Thread do frontend: buffer circular → (taxa do dispositivo → 16 kHz) → Silero VAD (blocos de 512)
→ segmentador → eventos para o pipeline.

Roda bem mais rápido que o tempo real (Silero em ONNX; custo medido em `ddsd benchmark --parts
frontend`) e nunca espera ASR nem classificadores: entrega os eventos por callback, e o pipeline os
enfileira.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable

import numpy as np

from ddsd.audio.resampler import StreamResampler
from ddsd.audio.ring_buffer import AudioRingBuffer
from ddsd.observability.logger import get_logger
from ddsd.observability.status import StatusBoard
from ddsd.realtime.config import MODEL_RATE, VAD_CHUNK, PipelineConfig
from ddsd.realtime.domain import CloseReason
from ddsd.vad.segmenter import Event, UtteranceSegmenter
from ddsd.vad.silero import SileroVad

log = get_logger("frontend")

#: Medidor de entrada do painel: baldes de 100 ms; 24 baldes de histórico (2,4 s); pico do último 1 s.
LEVEL_BUCKET_S = 0.1
LEVEL_HISTORY = 24
PEAK_BUCKETS = 10


def _to_dbfs(amplitude: float) -> float:
    return 20 * float(np.log10(amplitude)) if amplitude > 0 else -120.0

#: Sem dados da captura por este tempo = dispositivo parado (desconectado ou dormindo).
STALL_SECONDS = 2.0


class AudioFrontend(threading.Thread):
    """Processamento contínuo do áudio capturado, numa thread dedicada."""

    def __init__(self, cfg: PipelineConfig, ring: AudioRingBuffer, input_rate: Callable[[], int],
                 on_events: Callable[[list[Event]], None], status: StatusBoard,
                 on_stall: Callable[[], None] | None = None) -> None:
        super().__init__(name="frontend", daemon=True)
        self.cfg = cfg
        self.ring = ring
        self._input_rate = input_rate
        self._on_events = on_events
        self._on_stall = on_stall
        self.status = status
        self._stop = threading.Event()
        self._vad = SileroVad()
        self._segmenter = UtteranceSegmenter(cfg.vad, cfg.segmentation, chunk_samples=VAD_CHUNK)
        self._rate = 0
        self._to_model: StreamResampler | None = None  # taxa do dispositivo → 16 kHz
        self._pending = np.zeros(0, dtype=np.float32)
        self._abs16k = 0  # amostra absoluta (16 kHz) do início de `_pending`
        self._last_data = time.monotonic()
        self._silent_since: float | None = None
        self._silence_warned = False
        #: Pausa pedida pelo painel, aplicada pela thread do frontend entre blocos.
        self._paused = False
        self._pause_applied = False
        self._bucket_rms = self._bucket_peak = 0.0
        self._bucket_started = time.monotonic()
        self._level_history: deque[float] = deque(maxlen=LEVEL_HISTORY)
        self._peak_history: deque[float] = deque(maxlen=PEAK_BUCKETS)
        self.error: BaseException | None = None

    # --------------------------------------------------------------- controle
    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------- controles
    def set_paused(self, paused: bool) -> None:
        """Pausa (microfone fechado): a fala aberta é fechada e decidida; sem áudio, nada é reaberto."""
        self._paused = bool(paused)

    def run(self) -> None:
        try:
            while not self._stop.is_set():
                block = self.ring.read(timeout=0.1)
                now = time.monotonic()
                self._apply_pause(now)
                if self._paused:
                    continue  # pausado: o que sobrou no buffer é descartado e o travamento não é vigiado
                if block is None:
                    self._check_stall(now)
                    continue
                self._last_data = now
                if block.lost_before:
                    self._handle_gap(block.lost_before, now)
                self._process(block.samples, block.captured_at, now)
        except BaseException as exc:  # noqa: BLE001 - falha do frontend precisa chegar ao pipeline
            self.error = exc
            log.error("frontend falhou: %r", exc)
            self._emit(self._segmenter.close(CloseReason.ERROR, time.monotonic()))
        finally:
            self._emit(self._segmenter.close(CloseReason.SHUTDOWN, time.monotonic()))

    # ---------------------------------------------------------------- partes
    def _emit(self, events: list[Event]) -> None:
        if events:
            self._on_events(events)

    def _apply_pause(self, now: float) -> None:
        if self._paused and not self._pause_applied:
            self._pause_applied = True
            self._emit(self._segmenter.close(CloseReason.PAUSE, now))
            self.status.set("vad", estado="silêncio", prob=0.0, fala=None, duracao_s=0.0, silencio_ms=0)
        elif not self._paused and self._pause_applied:
            # Retomada: o áudio novo não continua o anterior; estados recorrentes recomeçam.
            self._pause_applied = False
            self._last_data = now
            self._pending = np.zeros(0, dtype=np.float32)
            self._vad.reset()
            if self._to_model is not None:
                self._to_model.reset()

    def _ensure_resampler(self) -> None:
        rate = self._input_rate()
        if rate != self._rate:
            if self._rate:
                log.warn("taxa de entrada mudou de %d para %d Hz; reamostrador reiniciado", self._rate, rate)
            self._rate = rate
            self._to_model = StreamResampler(rate, MODEL_RATE)

    def _handle_gap(self, lost_input_samples: int, now: float) -> None:
        """Amostras perdidas no buffer: avança o relógio de 16 kHz e reinicia estados (sem esconder a lacuna)."""
        lost16k = round(lost_input_samples * MODEL_RATE / max(self._rate, 1))
        log.error("lacuna de %.2f s na captura (buffer circular transbordou)", lost_input_samples / max(self._rate, 1))
        self.status.incr("captura.amostras_perdidas", lost_input_samples)
        self._abs16k += len(self._pending) + lost16k
        self._pending = np.zeros(0, dtype=np.float32)
        self._vad.reset()
        if self._to_model is not None:
            self._to_model.reset()

    def _check_stall(self, now: float) -> None:
        if now - self._last_data > STALL_SECONDS and self._on_stall is not None:
            log.warn("nenhum áudio há %.1f s; tentando reabrir o dispositivo", now - self._last_data)
            self._last_data = now  # evita reabrir em laço
            self._on_stall()

    def _track_level(self, samples: np.ndarray, now: float) -> None:
        rms = float(np.sqrt(np.mean(np.square(samples)))) if len(samples) else 0.0
        dbfs = 20 * np.log10(rms) if rms > 0 else -120.0
        # Medidor do painel: pico e nível em baldes de ~100 ms (histórico curto, publicado como tupla).
        self._bucket_rms = max(self._bucket_rms, rms)
        self._bucket_peak = max(self._bucket_peak, float(np.max(np.abs(samples))) if len(samples) else 0.0)
        if now - self._bucket_started >= LEVEL_BUCKET_S:
            self._level_history.append(_to_dbfs(self._bucket_rms))
            self._peak_history.append(self._bucket_peak)
            self._bucket_rms, self._bucket_peak, self._bucket_started = 0.0, 0.0, now
            self.status.set("captura", nivel_hist=tuple(self._level_history),
                            pico_dbfs=round(_to_dbfs(max(self._peak_history)), 1))
        self.status.set("captura", nivel_dbfs=round(dbfs, 1))
        if dbfs < self.cfg.audio.silent_input_dbfs:
            self._silent_since = self._silent_since or now
            if now - self._silent_since > self.cfg.audio.silent_input_s and not self._silence_warned:
                log.warn("entrada digitalmente silenciosa há %.0f s: verifique a permissão de microfone do "
                         "terminal (Ajustes > Privacidade > Microfone) ou o dispositivo", now - self._silent_since)
                self._silence_warned = True
        else:
            self._silent_since = None
            self._silence_warned = False

    def _process(self, samples: np.ndarray, captured_at: float, now: float) -> None:
        self._ensure_resampler()
        self._track_level(samples, now)
        assert self._to_model is not None
        at16k = self._to_model.process(samples)
        if len(at16k) == 0:
            return
        self._pending = np.concatenate([self._pending, at16k])
        events: list[Event] = []
        n_chunks = len(self._pending) // VAD_CHUNK
        for i in range(n_chunks):
            chunk = self._pending[i * VAD_CHUNK : (i + 1) * VAD_CHUNK]
            probability = self._vad.probability(chunk)
            events += self._segmenter.feed(chunk, probability, self._abs16k + i * VAD_CHUNK, captured_at, now)
        consumed = n_chunks * VAD_CHUNK
        self._pending = self._pending[consumed:].copy()
        self._abs16k += consumed
        seg = self._segmenter.status()
        self.status.set("vad", estado=seg.state.value, prob=round(seg.probability, 3),
                        fala=None if seg.utterance is None else f"U{seg.utterance:04d}",
                        duracao_s=round(seg.duration_s, 2), silencio_ms=round(seg.silence_ms))
        self.status.set("frontend", atraso_ms=round((time.monotonic() - captured_at) * 1000, 1),
                        audio_processado_s=round(self._abs16k / MODEL_RATE, 1))
        self._emit(events)
