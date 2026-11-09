"""Sorteio das reamostragens do bootstrap pareado: por item ou por frase.

Na voz do autor, as mesmas frases foram lidas em vários cenários, e os erros de um modelo se repetem na mesma
frase. Sortear os itens como se fossem independentes daria intervalos estreitos demais. Nesse conjunto, o
bootstrap sorteia as frases, e as leituras de cada frase sorteada entram juntas. Nos conjuntos em que cada
frase aparece uma vez, o sorteio continua por item.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence

import numpy as np

#: Conjuntos em que a mesma frase foi lida mais de uma vez (a voz do autor: as mesmas frases nos 4 cenários).
REPEATED_SENTENCE_DATASETS = frozenset({"gravacoes"})


def group_field(dataset: str) -> str | None:
    """Campo do registro que junta as leituras da mesma frase, ou None se cada frase aparece uma vez."""
    return "reference" if dataset in REPEATED_SENTENCE_DATASETS else None


def unit_name(field: str | None) -> str:
    """Unidade sorteada, para os relatórios: "item" ou "frase"."""
    return "item" if field is None else "frase"


def make_resampler(n: int, groups: Sequence[Hashable] | None = None
                   ) -> Callable[[np.random.Generator], np.ndarray]:
    """Devolve uma função que sorteia os índices de uma reamostragem.

    Sem `groups`, sorteia n itens com reposição. Com `groups` (um rótulo por item), sorteia os grupos com
    reposição e leva todos os itens de cada grupo sorteado. O tamanho da reamostragem varia com os grupos.
    """
    if groups is None:
        return lambda rng: rng.integers(0, n, n)
    if len(groups) != n:
        raise ValueError(f"{len(groups)} grupos para {n} itens")
    members: dict[Hashable, list[int]] = {}
    for i, g in enumerate(groups):  # ordem da primeira aparição: o sorteio é reprodutível com a mesma semente
        members.setdefault(g, []).append(i)
    blocks = [np.asarray(v) for v in members.values()]

    def draw(rng: np.random.Generator) -> np.ndarray:
        pick = rng.integers(0, len(blocks), len(blocks))
        return np.concatenate([blocks[k] for k in pick])

    return draw
