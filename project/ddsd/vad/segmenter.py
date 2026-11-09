"""Segmentação em falas a partir das probabilidades do VAD.

Máquina de estados por bloco de 32 ms (512 amostras a 16 kHz)

Regras:
- Pre-roll: a fala começa `pre_roll_ms` antes do início detectado, para não
  cortar o primeiro fonema.
- Pós-roll: da cauda de silêncio (hangover) ficam só `post_roll_ms` no áudio
  final, o que reduz alucinação do Whisper em silêncio longo.
- Uma tarefa por fala: o áudio da fala completa vai ao Whisper quando ela
  fecha, e a decisão é sempre a da fala completa.
- Sem perda de áudio: com o limite máximo, a continuação começa na amostra
  seguinte. Uma lacuna no fluxo de entrada (overrun do buffer de captura) fecha
  a fala aberta com o motivo `erro`.
"""

from __future__ import annotations

import bisect
import enum
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from ddsd.realtime.config import MODEL_RATE, SegmentationConfig, VadConfig
from ddsd.realtime.domain import AudioSpan, AudioTask, CloseReason, TaskId, UtteranceClosed, UtteranceOpened

Event = UtteranceOpened | UtteranceClosed


class SegmenterState(enum.StrEnum):
    SILENCE = "silêncio"
    CANDIDATE = "candidato"
    SPEECH = "fala"


@dataclass(frozen=True, slots=True)
class SegmenterStatus:
    state: SegmenterState
    probability: float
    utterance: int | None
    duration_s: float
    silence_ms: float


class UtteranceSegmenter:
    """Recebe (bloco, probabilidade) em ordem e devolve eventos de abertura e fechamento de fala."""

    def __init__(self, vad: VadConfig, segmentation: SegmentationConfig, *, chunk_samples: int = 512,
                 rate: int = MODEL_RATE, first_utterance: int = 1) -> None:
        chunk_ms = 1000.0 * chunk_samples / rate
        self._on, self._off = vad.threshold_on, vad.threshold_off
        self._min_speech_chunks = max(1, math.ceil(vad.min_speech_ms / chunk_ms))
        self._min_silence_chunks = max(1, math.ceil(vad.min_silence_ms / chunk_ms))
        self._pre_roll = round(vad.pre_roll_ms * rate / 1000)
        self._post_roll = round(vad.post_roll_ms * rate / 1000)
        self._max = round(segmentation.max_utterance_s * rate)
        self._chunk = chunk_samples
        self._next_id = first_utterance
        self._expected: int | None = None
        self._history: deque[np.ndarray] = deque(maxlen=math.ceil(self._pre_roll / chunk_samples) + 1)
        self._history_end = 0
        self._candidate: list[tuple[int, np.ndarray, float]] = []
        self._buffer = np.zeros(self._max, dtype=np.float32)
        self._reset_utterance()
        self._state = SegmenterState.SILENCE
        self._last_prob = 0.0

    # ------------------------------------------------------------------ estado
    def _reset_utterance(self) -> None:
        self._utterance: int | None = None
        self._continuation_of: int | None = None
        self._start_abs = 0
        self._length = 0
        self._chunk_ends: list[int] = []
        self._chunk_times: list[float] = []
        self._silence_run = 0
        self._last_speech_end = 0
        self._last_speech_time = 0.0

    def status(self) -> SegmenterStatus:
        return SegmenterStatus(self._state, self._last_prob, self._utterance, self._length / MODEL_RATE,
                               self._silence_run * 1000.0 * self._chunk / MODEL_RATE)

    # ---------------------------------------------------------------- entrada
    def feed(self, chunk: np.ndarray, probability: float, start: int, captured_at: float,
             now: float) -> list[Event]:
        """Processa um bloco que começa na amostra absoluta `start`."""
        events: list[Event] = []
        if self._expected is not None and start != self._expected:
            events += self._on_gap(now)
        self._expected = start + len(chunk)
        self._last_prob = probability
        speech = probability >= self._off

        if self._state is SegmenterState.SPEECH:
            if self._length + len(chunk) > self._max:
                previous = self._utterance
                events += self._close(CloseReason.MAX_DURATION, now)
                if speech:
                    events.append(self._open(start, [chunk], [captured_at], now, continuation_of=previous))
                    self._track_speech(probability, start + len(chunk), captured_at)
                else:
                    self._remember(chunk, start)
                return events
            self._append(chunk, captured_at)
            self._track_speech(probability, start + len(chunk), captured_at)
            if self._silence_run >= self._min_silence_chunks:
                events += self._close(CloseReason.END_OF_SPEECH, now)
            return events

        if self._state is SegmenterState.CANDIDATE:
            if speech:
                self._candidate.append((start, chunk, captured_at))
                if len(self._candidate) >= self._min_speech_chunks:
                    events.append(self._open_from_candidate(now))
            else:
                for c_start, c_chunk, _ in self._candidate:
                    self._remember(c_chunk, c_start)
                self._candidate = []
                self._remember(chunk, start)
                self._state = SegmenterState.SILENCE
            return events

        if probability >= self._on:
            self._candidate = [(start, chunk, captured_at)]
            self._state = SegmenterState.CANDIDATE
            if self._min_speech_chunks == 1:
                events.append(self._open_from_candidate(now))
        else:
            self._remember(chunk, start)
        return events

    def close(self, reason: CloseReason, now: float) -> list[Event]:
        """Fecha a fala aberta (encerramento ou erro); candidatos não confirmados são ignorados."""
        events = self._close(reason, now) if self._state is SegmenterState.SPEECH else []
        self._candidate = []
        self._state = SegmenterState.SILENCE
        return events

    # ----------------------------------------------------------------- partes
    def _remember(self, chunk: np.ndarray, start: int) -> None:
        if self._history and self._history_end != start:
            self._history.clear()
        self._history.append(chunk)
        self._history_end = start + len(chunk)

    def _on_gap(self, now: float) -> list[Event]:
        events = self._close(CloseReason.ERROR, now) if self._state is SegmenterState.SPEECH else []
        self._history.clear()
        self._candidate = []
        self._state = SegmenterState.SILENCE
        return events

    def _open_from_candidate(self, now: float) -> UtteranceOpened:
        first_start = self._candidate[0][0]
        pre: list[np.ndarray] = []
        if self._history and self._history_end == first_start and self._pre_roll > 0:
            tail = np.concatenate(list(self._history))[-self._pre_roll :]
            pre = [tail]
        chunks = pre + [c for _, c, _ in self._candidate]
        times = [self._candidate[0][2]] * len(pre) + [t for _, _, t in self._candidate]
        start = first_start - (len(pre[0]) if pre else 0)
        event = self._open(start, chunks, times, now)
        last_start, last_chunk, last_time = self._candidate[-1]
        self._last_speech_end = last_start + len(last_chunk) - self._start_abs
        self._last_speech_time = last_time
        self._candidate = []
        self._history.clear()
        return event

    def _open(self, start: int, chunks: list[np.ndarray], times: list[float], now: float,
              continuation_of: int | None = None) -> UtteranceOpened:
        self._reset_utterance()
        self._utterance = self._next_id
        self._next_id += 1
        self._continuation_of = continuation_of
        self._start_abs = start
        for chunk, captured_at in zip(chunks, times, strict=True):
            self._append(chunk, captured_at)
        self._state = SegmenterState.SPEECH
        return UtteranceOpened(self._utterance, start, now, continuation_of)

    def _append(self, chunk: np.ndarray, captured_at: float) -> None:
        self._buffer[self._length : self._length + len(chunk)] = chunk
        self._length += len(chunk)
        self._chunk_ends.append(self._length)
        self._chunk_times.append(captured_at)

    def _track_speech(self, probability: float, end_abs: int, captured_at: float) -> None:
        if probability >= self._off:
            self._silence_run = 0
            self._last_speech_end = end_abs - self._start_abs
            self._last_speech_time = captured_at
        else:
            self._silence_run += 1

    def _time_at(self, rel_end: int) -> float:
        """Instante de captura do bloco que contém a amostra relativa `rel_end - 1`."""
        index = bisect.bisect_left(self._chunk_ends, rel_end)
        return self._chunk_times[min(index, len(self._chunk_times) - 1)]

    def _close(self, reason: CloseReason, now: float) -> list[Event]:
        if reason is CloseReason.END_OF_SPEECH:
            length = min(self._length, self._last_speech_end + self._post_roll)
        else:
            length = self._length
        length = max(length, 1)
        span = AudioSpan(self._start_abs, self._start_abs + length)
        task = AudioTask.create(TaskId(self._utterance), span, self._buffer[:length],
                                self._time_at(length), now, close_reason=reason,
                                continuation_of=self._continuation_of)
        speech_end = self._last_speech_time if reason is CloseReason.END_OF_SPEECH else task.captured_at
        closed = UtteranceClosed(task=task, speech_ended_at=speech_end)
        self._reset_utterance()
        self._state = SegmenterState.SILENCE
        return [closed]
