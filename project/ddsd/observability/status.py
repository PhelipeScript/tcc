"""Estado dos componentes para o painel, separado do logger.

Os produtores (captura, frontend, ASR, workers, decisor) atualizam o
`StatusBoard` com operações O(1) sob um lock curto. O painel lê um `snapshot()`
imutável quatro vezes por segundo e nunca bloqueia quem produz.
"""

from __future__ import annotations

import copy
import threading
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FeedItem:
    """Linha de um fluxo do painel (transcrições, resultados, decisões)."""

    at: float
    task: str
    kind: str  # "final" (a tarefa da fala completa)
    text: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class UtteranceView:
    """Uma fala, da abertura à decisão (para o painel)."""

    utterance: str             # "U0007"
    start_s: float             # início da fala, em segundos desde o início da captura
    opened_wall: float         # hora da abertura (time.time), para exibir
    #: capturando | transcrevendo (a final) | analisando | decidida
    state: str = "capturando"
    classifier: str | None = None  # fixado na abertura da fala
    duration_s: float | None = None
    final_text: str | None = None
    final_normalized: str | None = None
    label: str | None = None
    reason: str | None = None
    p_dd: float | None = None
    threshold: float | None = None
    reliability_score: float | None = None
    reliability_label: str | None = None
    reliability_alert: bool = False
    latency_s: float | None = None


@dataclass(frozen=True)
class StatusSnapshot:
    started_at: float
    now: float
    components: dict[str, dict[str, Any]]
    counters: dict[str, int]
    latencies: dict[str, tuple[int, float, float, float]]  # nome -> (n, p50, p95, último)
    feeds: dict[str, list[FeedItem]]
    #: Falas mais recentes, da mais antiga à atual (a atual é a última).
    utterances: list[UtteranceView] = field(default_factory=list)


class StatusBoard:
    """Estado compartilhado e thread-safe do pipeline."""

    #: Falas guardadas para o painel: a atual e as últimas decididas.
    MAX_UTTERANCES = 8

    def __init__(self, feed_size: int = 12, latency_window: int = 500) -> None:
        self._lock = threading.Lock()
        self._utterances: OrderedDict[str, UtteranceView] = OrderedDict()
        self._started = time.monotonic()
        self._components: dict[str, dict[str, Any]] = defaultdict(dict)
        self._counters: dict[str, int] = defaultdict(int)
        self._latencies: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=latency_window))
        self._feeds: dict[str, deque[FeedItem]] = defaultdict(lambda: deque(maxlen=feed_size))

    def set(self, component: str, **fields: Any) -> None:
        with self._lock:
            self._components[component].update(fields)

    def incr(self, counter: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[counter] += amount

    def observe(self, name: str, seconds: float) -> None:
        with self._lock:
            self._latencies[name].append(seconds)

    def push(self, feed: str, item: FeedItem) -> None:
        with self._lock:
            self._feeds[feed].append(item)

    def get(self, component: str, key: str, default: Any = None) -> Any:
        """Um campo de um componente, sem copiar o estado inteiro (uso do monitor)."""
        with self._lock:
            return self._components.get(component, {}).get(key, default)

    # ------------------------------------------------------------ falas
    def utterance_opened(self, utterance: str, start_s: float, opened_wall: float) -> None:
        with self._lock:
            self._utterances[utterance] = UtteranceView(utterance, start_s, opened_wall)
            while len(self._utterances) > self.MAX_UTTERANCES:
                self._utterances.popitem(last=False)

    def utterance_update(self, utterance: str, **fields: Any) -> None:
        """Atualiza os campos de uma fala conhecida (falas já fora da janela do painel são ignoradas)."""
        with self._lock:
            view = self._utterances.get(utterance)
            if view is not None:
                for key, value in fields.items():
                    setattr(view, key, value)

    def snapshot(self) -> StatusSnapshot:
        with self._lock:
            components = copy.deepcopy(dict(self._components))
            counters = dict(self._counters)
            samples = {name: list(values) for name, values in self._latencies.items() if values}
            feeds = {name: list(items) for name, items in self._feeds.items()}
            utterances = copy.deepcopy(list(self._utterances.values()))
        latencies = {name: (len(v), float(np.percentile(v, 50)), float(np.percentile(v, 95)), v[-1])
                     for name, v in samples.items()}
        return StatusSnapshot(self._started, time.monotonic(), components, counters, latencies, feeds,
                              utterances)
