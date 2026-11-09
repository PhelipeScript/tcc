"""Fila de tarefas thread-safe, em ordem de chegada (FIFO), sem descarte.

Usada pela fila do ASR e pelas filas de fuzzy e do classificador. Cada fala gera uma única tarefa (a
fala completa), então a quantidade de itens é limitada pela própria fala humana: nada é descartado.
`close()` encerra a entrada; os consumidores esvaziam o que restou e então recebem `QueueClosed`.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


class QueueClosed(Exception):
    """A fila foi fechada e esvaziada."""


@dataclass(frozen=True, slots=True)
class QueueStats:
    pending: int
    enqueued: int


class JobQueue(Generic[T]):
    """Fila FIFO bloqueante com fechamento."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: deque[T] = deque()
        self._cond = threading.Condition()
        self._closed = False
        self._enqueued = 0

    def put(self, item: T) -> None:
        with self._cond:
            if self._closed:
                raise QueueClosed(self.name)
            self._enqueued += 1
            self._items.append(item)
            self._cond.notify()

    def get(self, timeout: float | None = None) -> T | None:
        """Próxima tarefa. `None` se o tempo esgotar; `QueueClosed` se a fila estiver fechada e vazia."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._items or self._closed, timeout):
                return None
            if self._items:
                return self._items.popleft()
            raise QueueClosed(self.name)

    def close(self) -> None:
        """Não aceita novas tarefas; os consumidores esvaziam o que restou e recebem `QueueClosed`."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def stats(self) -> QueueStats:
        with self._cond:
            return QueueStats(len(self._items), self._enqueued)
