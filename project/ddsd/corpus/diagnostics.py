"""Diagnósticos de atalhos no corpus: quanto o comprimento, a fonte ou a origem predizem a classe.

Todos devolvem AUC com DD como classe positiva:
- 0,5: o atributo não carrega informação sobre a classe;
- longe de 0,5, para qualquer lado: o classificador pode usar o atributo como atalho.

`informativeness = max(AUC, 1 - AUC)` resume os dois lados num número só.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence

import numpy as np


def auc_from_scores(scores: Sequence[float], is_positive: Sequence[bool]) -> float:
    """AUC de Mann-Whitney (empates valem 1/2). Devolve NaN se faltar uma das classes."""
    scores_arr = np.asarray(scores, dtype=float)
    positive = np.asarray(is_positive, dtype=bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores_arr, kind="mergesort")
    ranks = np.empty(len(scores_arr), dtype=float)
    sorted_scores = scores_arr[order]
    # Postos médios para empates.
    _, first_index, counts = np.unique(sorted_scores, return_index=True, return_counts=True)
    average_rank = first_index + (counts + 1) / 2.0
    ranks[order] = np.repeat(average_rank, counts)
    rank_sum = ranks[positive].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def informativeness(auc: float) -> float:
    """Distância de 0,5 dobrada para um lado só: max(AUC, 1 - AUC)."""
    return float("nan") if np.isnan(auc) else max(auc, 1.0 - auc)


def length_auc(labels: Sequence[str], words: Sequence[int]) -> float:
    """AUC de "quanto mais curta, mais DD" (score = -palavras)."""
    return auc_from_scores([-w for w in words], [lab == "DD" for lab in labels])


def categorical_auc_within_bins(labels: Sequence[str], values: Sequence[str],
                                bins: Sequence[str]) -> tuple[float, dict[str, float]]:
    """AUC de prever a classe pelo atributo categórico, dentro de cada faixa.

    O score de cada item é P(DD | valor, faixa), estimado na própria faixa. Isso
    dá o melhor desempenho que um classificador poderia tirar só do atributo, ou
    seja, a medida in-sample do atalho. O total é a média das AUCs por faixa,
    ponderada pelo tamanho da faixa (faixas com uma classe só ficam de fora).
    """
    by_bin: dict[str, list[int]] = defaultdict(list)
    for index, b in enumerate(bins):
        by_bin[b].append(index)
    per_bin: dict[str, float] = {}
    weighted, weight = 0.0, 0
    for b, indices in by_bin.items():
        dd = Counter(values[i] for i in indices if labels[i] == "DD")
        total = Counter(values[i] for i in indices)
        scores = [dd[values[i]] / total[values[i]] for i in indices]
        auc = auc_from_scores(scores, [labels[i] == "DD" for i in indices])
        per_bin[b] = auc
        if not np.isnan(auc):
            weighted += auc * len(indices)
            weight += len(indices)
    return (weighted / weight if weight else float("nan")), per_bin


def length_histogram(labels: Sequence[str], bins: Sequence[str],
                     order: Sequence[str]) -> list[tuple[str, int, int]]:
    """Contagem (faixa, DD, NDD) na ordem das faixas."""
    counts = Counter(zip(bins, labels, strict=True))
    return [(b, counts[(b, "DD")], counts[(b, "NDD")]) for b in order]
