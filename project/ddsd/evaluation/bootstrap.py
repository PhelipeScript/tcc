"""Bootstrap pareado entre modelos no mesmo teste (IC 95% e P(A > B) por métrica).

A cada reamostragem, os mesmos índices do teste são usados para os dois
modelos comparados. Isso só é válido se as predições estiverem na mesma ordem;
`check_alignment` confere o texto e o rótulo linha a linha e aborta se houver
desalinhamento, em vez de produzir um IC inválido em silêncio.

Com várias sementes por modelo, a predição usada é a da semente escolhida pela
F1 de validação (a mesma regra do pipeline). A variação entre sementes é
reportada à parte (média ± desvio).
"""

from __future__ import annotations

import itertools
from collections.abc import Hashable, Sequence
from pathlib import Path

import numpy as np

from ddsd.common.io import read_jsonl
from ddsd.evaluation.resampling import make_resampler
from ddsd.training.config import LABEL2ID
from ddsd.training.metrics import classification_metrics

BOOTSTRAP_METRICS: tuple[str, ...] = ("accuracy", "f1_macro", "roc_auc", "eer")
#: Métricas em que valor menor é melhor (EER).
LOWER_IS_BETTER = frozenset({"eer"})


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    rows = list(read_jsonl(path))
    return {"text": np.array([r.get("text", "") for r in rows], dtype=object),
            "y_true": np.array([LABEL2ID[r["label"]] for r in rows]),
            "y_pred": np.array([LABEL2ID[r["pred"]] for r in rows]),
            "y_score": np.array([r["score_dd"] for r in rows], dtype=float)}


def check_alignment(predictions: dict[str, dict[str, np.ndarray]]) -> None:
    names = list(predictions)
    ref = predictions[names[0]]
    for name in names[1:]:
        other = predictions[name]
        if len(other["y_true"]) != len(ref["y_true"]):
            raise ValueError(f"{name}: {len(other['y_true'])} exemplos contra {len(ref['y_true'])} em {names[0]}")
        if not np.array_equal(other["y_true"], ref["y_true"]) or not np.array_equal(other["text"], ref["text"]):
            raise ValueError(f"predições de {name} desalinhadas com {names[0]} - bootstrap pareado inválido")


def paired_bootstrap(a: dict[str, np.ndarray], b: dict[str, np.ndarray], *, n_bootstrap: int = 2000,
                     seed: int = 42, groups: Sequence[Hashable] | None = None) -> dict[str, dict[str, float]]:
    """Diferença métrica(A) - métrica(B): média, IC 95% e probabilidade de A ser melhor.

    Com `groups`, sorteia as frases em vez dos itens (`ddsd.evaluation.resampling`).
    """
    rng = np.random.default_rng(seed)
    y = a["y_true"]
    draw = make_resampler(len(y), groups)
    diffs = {k: np.empty(n_bootstrap) for k in BOOTSTRAP_METRICS}
    for i in range(n_bootstrap):
        idx = draw(rng)
        ma = classification_metrics(y[idx], a["y_pred"][idx], a["y_score"][idx])
        mb = classification_metrics(y[idx], b["y_pred"][idx], b["y_score"][idx])
        for key in BOOTSTRAP_METRICS:
            diffs[key][i] = ma[key] - mb[key]
    out = {}
    for key, values in diffs.items():
        better = values < 0 if key in LOWER_IS_BETTER else values > 0
        out[key] = {"diferenca_media": float(np.mean(values)), "ic95_inf": float(np.percentile(values, 2.5)),
                    "ic95_sup": float(np.percentile(values, 97.5)), "p_a_melhor": float(np.mean(better))}
    return out


def pairwise(predictions: dict[str, dict[str, np.ndarray]], n_bootstrap: int = 2000,
             groups: Sequence[Hashable] | None = None) -> dict[str, dict]:
    check_alignment(predictions)
    return {f"{a} vs {b}": paired_bootstrap(predictions[a], predictions[b], n_bootstrap=n_bootstrap, groups=groups)
            for a, b in itertools.combinations(predictions, 2)}
