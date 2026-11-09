"""Partição treino/validação/teste estratificada, determinística e sem texto em comum.

Cada estrato (classe|fonte|faixa) é embaralhado com semente fixa e dividido nas
proporções configuradas pelo método do maior resto, de modo que as proporções
valham dentro de cada estrato. Estratos menores que `min_stratum` caem para
classe|faixa e, se ainda forem pequenos, para a classe. Assim nenhum grupo
minúsculo distorce a divisão.

Como o corpus já foi deduplicado (inclusive por similaridade), os textos são
únicos, e a disjunção entre partições vale por construção. Mesmo assim, ela é
verificada em `verify_disjoint`.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Sequence

SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")


def stratum_keys(labels: Sequence[str], sources: Sequence[str], bins: Sequence[str],
                 min_stratum: int) -> list[str]:
    """Chave de estrato de cada item, com recuo para estratos mais grossos quando pequenos."""
    fine = [f"{lab}|{src}|{b}" for lab, src, b in zip(labels, sources, bins, strict=True)]
    fine_counts = Counter(fine)
    keys = [k if fine_counts[k] >= min_stratum else f"{lab}|*|{b}"
            for k, lab, b in zip(fine, labels, bins, strict=True)]
    mid_counts = Counter(keys)
    return [k if (not k.split("|")[1] == "*" or mid_counts[k] >= min_stratum) else f"{lab}|*|*"
            for k, lab in zip(keys, labels, strict=True)]


def _allocate(n: int, ratios: Sequence[float]) -> list[int]:
    """Divide `n` itens nas proporções, pelo maior resto (desempate pela ordem da partição)."""
    exact = [n * r for r in ratios]
    counts = [int(x) for x in exact]
    for i in sorted(range(len(ratios)), key=lambda j: (-(exact[j] - counts[j]), j))[: n - sum(counts)]:
        counts[i] += 1
    return counts


def stratified_split(keys: Sequence[str], ratios: Sequence[float], seed: int) -> dict[str, list[int]]:
    """Índices de cada partição, ordenados pela ordem original."""
    by_key: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        by_key[key].append(index)
    parts: dict[str, list[int]] = {name: [] for name in SPLIT_NAMES}
    for key in sorted(by_key):
        indices = list(by_key[key])
        random.Random(f"{seed}|{key}").shuffle(indices)
        start = 0
        for name, count in zip(SPLIT_NAMES, _allocate(len(indices), ratios), strict=True):
            parts[name].extend(indices[start : start + count])
            start += count
    return {name: sorted(ix) for name, ix in parts.items()}


def verify_disjoint(texts: Sequence[str], parts: dict[str, list[int]]) -> None:
    """Levanta `AssertionError` se algum texto aparecer em mais de uma partição."""
    seen: dict[str, str] = {}
    for name in SPLIT_NAMES:
        for index in parts[name]:
            other = seen.setdefault(texts[index], name)
            if other != name:
                raise AssertionError(f"texto em {other} e {name}: {texts[index]!r}")
