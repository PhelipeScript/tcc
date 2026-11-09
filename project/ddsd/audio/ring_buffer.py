"""Buffer circular de captura: desacopla o callback do PortAudio do resto do pipeline.

O callback só copia o bloco para o buffer pré-alocado.
Todo o processamento roda na thread do frontend, que lê o que estiver pendente.

Se o leitor ficar parado por mais que a capacidade, o escritor sobrescreve as
amostras mais antigas não lidas. Isso nunca é silencioso: as amostras
perdidas são contadas em `overrun_samples`, e o leitor recebe o deslocamento
absoluto real do bloco seguinte, que expõe o buraco.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RingRead:
    """Bloco lido: amostras, deslocamento absoluto da primeira e instante de captura da última."""

    samples: np.ndarray
    start: int
    captured_at: float
    lost_before: int


@dataclass(frozen=True, slots=True)
class RingStats:
    capacity: int
    written: int
    read: int
    unread: int
    overrun_samples: int

    @property
    def fill_ratio(self) -> float:
        return self.unread / self.capacity if self.capacity else 0.0


class AudioRingBuffer:
    """Buffer circular mono `float32`, com um escritor (callback) e um leitor (frontend)."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacidade deve ser positiva")
        self._buffer = np.zeros(capacity, dtype=np.float32)
        self._capacity = capacity
        self._written = 0
        self._read = 0
        self._overrun = 0
        self._lost_pending = 0
        self._last_captured_at = 0.0
        self._lock = threading.Lock()
        self._ready = threading.Event()

    def write(self, block: np.ndarray, captured_at: float) -> None:
        """Copia `block` (1-D) para o buffer. Chamado pelo callback de áudio: rápido e sem log."""
        n = len(block)
        if n == 0:
            return
        with self._lock:
            if n > self._capacity:  # bloco maior que o buffer: só cabem as últimas amostras
                skipped = n - self._capacity
                self._written += skipped
                block = block[skipped:]
                n = self._capacity
            start = self._written % self._capacity
            first = min(n, self._capacity - start)
            self._buffer[start : start + first] = block[:first]
            if first < n:
                self._buffer[: n - first] = block[first:]
            self._written += n
            unread = self._written - self._read
            if unread > self._capacity:
                lost = unread - self._capacity
                self._read += lost
                self._overrun += lost
                self._lost_pending += lost
            self._last_captured_at = captured_at
        self._ready.set()

    def read(self, timeout: float | None = None) -> RingRead | None:
        """Tudo que estiver pendente, ou `None` se nada chegar em `timeout` segundos."""
        if not self._ready.wait(timeout):
            return None
        with self._lock:
            n = self._written - self._read
            self._ready.clear()
            if n == 0:
                return None
            start_abs = self._read
            start = start_abs % self._capacity
            first = min(n, self._capacity - start)
            out = np.empty(n, dtype=np.float32)
            out[:first] = self._buffer[start : start + first]
            if first < n:
                out[first:] = self._buffer[: n - first]
            self._read += n
            lost, self._lost_pending = self._lost_pending, 0
            return RingRead(samples=out, start=start_abs, captured_at=self._last_captured_at, lost_before=lost)

    def stats(self) -> RingStats:
        with self._lock:
            return RingStats(self._capacity, self._written, self._read, self._written - self._read, self._overrun)
