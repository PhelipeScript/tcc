"""Captura contínua do microfone com sounddevice (PortAudio).

O callback roda na thread de tempo real do CoreAudio e faz o mínimo: copia o
canal 0 para o `AudioRingBuffer`, anota o instante e conta flags de overflow.
Nada de log, alocação ou lock longo ali. Tudo mais (nível, silêncio,
reconexão) é verificado fora do callback, pela thread do frontend.

Dispositivo: índice, parte do nome (ex.: "Mac") ou vazio para o padrão do
sistema. A captura usa a **taxa nativa do dispositivo**; a conversão para
16 kHz (VAD/Whisper) é feita no frontend, com estado.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ddsd.audio.ring_buffer import AudioRingBuffer
from ddsd.observability.logger import get_logger

log = get_logger("captura")


class CaptureError(RuntimeError):
    """Falha ao abrir ou manter o dispositivo de entrada."""


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    samplerate: int
    channels: int
    is_default: bool


def list_input_devices() -> list[InputDevice]:
    import sounddevice as sd

    default_in = sd.default.device[0] if sd.default.device else -1
    devices = []
    for index, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] > 0:
            devices.append(InputDevice(index, info["name"], int(info["default_samplerate"]),
                                       int(info["max_input_channels"]), index == default_in))
    return devices


def resolve_device(spec: str) -> InputDevice:
    """Dispositivo pelo índice, por parte do nome (sem diferenciar maiúsculas) ou o padrão."""
    devices = list_input_devices()
    if not devices:
        raise CaptureError("nenhum dispositivo de entrada de áudio encontrado")
    spec = spec.strip()
    if not spec:
        default = next((d for d in devices if d.is_default), devices[0])
        return default
    if spec.isdigit():
        match = next((d for d in devices if d.index == int(spec)), None)
    else:
        match = next((d for d in devices if spec.lower() in d.name.lower()), None)
    if match is None:
        raise CaptureError(f"dispositivo {spec!r} não encontrado; disponíveis: {[d.name for d in devices]}")
    return match


class AudioCapture:
    """Stream de entrada mono que alimenta o buffer circular."""

    def __init__(self, ring: AudioRingBuffer, device: InputDevice, samplerate: int = 0,
                 latency: str = "high", blocksize: int = 0) -> None:
        self.ring = ring
        self.device = device
        self.samplerate = samplerate or device.samplerate
        self._latency = latency
        self._blocksize = blocksize
        self._stream: Any = None
        self._lock = threading.Lock()
        self.overflows = 0
        self.callbacks = 0
        self.last_callback_at = 0.0
        self.frames_captured = 0

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        # Thread de tempo real: só cópia e contadores.
        if status and status.input_overflow:
            self.overflows += 1
        now = time.monotonic()
        self.ring.write(indata[:, 0], now)
        self.callbacks += 1
        self.frames_captured += frames
        self.last_callback_at = now

    def start(self) -> None:
        import sounddevice as sd

        with self._lock:
            if self._stream is not None:
                return
            try:
                self._stream = sd.InputStream(device=self.device.index, channels=1, samplerate=self.samplerate,
                                              dtype="float32", latency=self._latency,
                                              blocksize=self._blocksize, callback=self._callback)
                self._stream.start()
            except Exception as exc:  # noqa: BLE001 - PortAudio levanta tipos variados
                self._stream = None
                raise CaptureError(f"não foi possível abrir {self.device.name!r} a {self.samplerate} Hz: {exc}") from exc
            self.last_callback_at = time.monotonic()
        log.info("captura iniciada: %s (%d Hz, latência %s, bloco %s)", self.device.name, self.samplerate,
                 self._latency, self._blocksize or "auto")

    def stop(self) -> None:
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:  # noqa: BLE001
                log.warn("erro ao fechar o stream: %s", exc)
            log.info("captura encerrada (%d callbacks, %d overflows)", self.callbacks, self.overflows)

    @property
    def active(self) -> bool:
        with self._lock:
            return self._stream is not None and bool(self._stream.active)

    def restart(self) -> None:
        """Reabre o stream. Reinicializa o PortAudio, que guarda a lista de dispositivos em cache."""
        import sounddevice as sd

        self.stop()
        sd._terminate()  # noqa: SLF001 - única forma pública de recarregar a lista de dispositivos
        sd._initialize()  # noqa: SLF001
        try:
            self.device = resolve_device(self.device.name)
        except CaptureError:
            log.warn("dispositivo %r sumiu; usando o padrão do sistema", self.device.name)
            self.device = resolve_device("")
        self.samplerate = self.device.samplerate
        self.start()
