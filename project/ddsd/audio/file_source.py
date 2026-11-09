"""Fonte de áudio a partir de gravações, para testes, avaliação e benchmarks.

Alimenta o mesmo `AudioRingBuffer` que a captura do microfone. Por isso
frontend, VAD, segmentação, ASR, fuzzy, classificadores e decisor rodam
exatamente como ao vivo. Ela não simula nada do pipeline, só substitui o
microfone por um arquivo real.

Modos:
- `realtime`: blocos no ritmo do relógio (latências comparáveis às do uso real);
- `fast`: o mais rápido possível, sem estourar o buffer (espera haver espaço).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from ddsd.audio.ring_buffer import AudioRingBuffer
from ddsd.observability.logger import get_logger

log = get_logger("replay")


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    """Lê o arquivo como mono `float32` na taxa original."""
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1).astype(np.float32), int(rate)


class FileAudioSource:
    """Escreve uma sequência de sinais no buffer circular, em ritmo real ou acelerado."""

    def __init__(self, ring: AudioRingBuffer, samplerate: int, *, realtime: bool = True,
                 block_ms: float = 10.0) -> None:
        self.ring = ring
        self.samplerate = samplerate
        self.realtime = realtime
        self._block = max(1, int(samplerate * block_ms / 1000))
        self._stop = threading.Event()
        self.samples_written = 0

    def play(self, signal: np.ndarray) -> None:
        """Escreve `signal` (já na taxa da fonte) no buffer; bloqueia até o fim ou `stop()`."""
        started = time.monotonic()
        base = self.samples_written
        for offset in range(0, len(signal), self._block):
            if self._stop.is_set():
                return
            block = signal[offset : offset + self._block]
            if self.realtime:
                due = started + (offset + len(block)) / self.samplerate
                delay = due - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            else:
                # Modo acelerado: não deixa o buffer passar de metade da capacidade.
                while self.ring.stats().unread > self.ring.stats().capacity // 2 and not self._stop.is_set():
                    time.sleep(0.002)
            self.ring.write(block, time.monotonic())
            self.samples_written = base + offset + len(block)

    def play_silence(self, seconds: float) -> None:
        self.play(np.zeros(int(seconds * self.samplerate), dtype=np.float32))

    def stop(self) -> None:
        self._stop.set()
