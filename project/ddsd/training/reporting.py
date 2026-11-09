"""Artefatos de uma execução de treino: métricas, gráficos, predições e manifesto."""

from __future__ import annotations

import platform
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from ddsd.common.io import write_json, write_jsonl
from ddsd.common.system import code_version
from ddsd.observability.logger import get_logger
from ddsd.training.config import LABEL_NAMES
from ddsd.training.data import SplitMetadata
from ddsd.training.metrics import classification_metrics, confusion

log = get_logger("treino.relatorio")

#: Métricas da tabela comparativa, na ordem de exibição.
REPORT_METRICS: tuple[str, ...] = ("accuracy", "f1_dd", "f1_macro", "roc_auc", "eer", "mcc")


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, run_dir: Path, split: str) -> None:
    matrix = confusion(y_true, y_pred)
    lines = ["verdadeiro/predito," + ",".join(LABEL_NAMES)]
    lines += [f"{LABEL_NAMES[i]}," + ",".join(str(int(v)) for v in matrix[i]) for i in range(2)]
    (run_dir / f"confusion_matrix_{split}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(4.2, 3.8))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(2), LABEL_NAMES)
    axis.set_yticks(range(2), LABEL_NAMES)
    axis.set_xlabel("predito")
    axis.set_ylabel("verdadeiro")
    axis.set_title(f"Matriz de confusão - {split}")
    for i in range(2):
        for j in range(2):
            axis.text(j, i, f"{int(matrix[i, j]):,}".replace(",", "."), ha="center", va="center",
                      color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    figure.tight_layout()
    figure.savefig(run_dir / f"confusion_matrix_{split}.png", dpi=150)
    plt.close(figure)


def save_curves(y_true: np.ndarray, y_score: np.ndarray, run_dir: Path, split: str, eer: float) -> None:
    """Curvas ROC e DET (falso positivo × falso negativo, a leitura natural de um sistema de ativação)."""
    if len(np.unique(y_true)) < 2:
        return
    from sklearn.metrics import roc_auc_score, roc_curve

    plt = _pyplot()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    figure, axis = plt.subplots(figsize=(4.4, 4.0))
    axis.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_score):.4f}")
    axis.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
    axis.set_xlabel("taxa de falsos positivos")
    axis.set_ylabel("taxa de verdadeiros positivos")
    axis.set_title(f"Curva ROC - {split}")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(run_dir / f"roc_curve_{split}.png", dpi=150)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(4.4, 4.0))
    axis.plot(fpr, 1.0 - tpr, label="DET")
    if not np.isnan(eer):
        axis.plot([eer], [eer], "o", color="crimson", label=f"EER = {eer:.4f}")
    axis.set_xlabel("falsos positivos (acordar sem querer)")
    axis.set_ylabel("falsos negativos (não acordar)")
    axis.set_title(f"Curva DET - {split}")
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(run_dir / f"det_curve_{split}.png", dpi=150)
    plt.close(figure)


def save_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray, meta: SplitMetadata,
                     run_dir: Path, split: str) -> None:
    """Uma linha por exemplo, **com o texto**, para a paridade do pipeline e a análise de erro."""
    rows = ({"text": meta.text[i], "label": LABEL_NAMES[int(y_true[i])], "pred": LABEL_NAMES[int(y_pred[i])],
             "score_dd": round(float(y_score[i]), 6), "source": meta.source[i], "origin": meta.origin[i],
             "bin": meta.bin[i], "correto": bool(y_true[i] == y_pred[i])} for i in range(len(y_true)))
    write_jsonl(run_dir / f"predictions_{split}.jsonl", rows)


def _grouped_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray,
                     groups: Sequence[str]) -> dict[str, Any]:
    groups_arr = np.asarray(groups)
    out: dict[str, Any] = {}
    for group in sorted(set(groups)):
        mask = groups_arr == group
        if len(np.unique(y_true[mask])) < 2:
            out[group] = {"n": int(mask.sum()), "observacao": "classe única; métricas não calculadas"}
            continue
        out[group] = {"n": int(mask.sum()), **classification_metrics(y_true[mask], y_pred[mask], y_score[mask])}
    return out


def save_grouped_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray, meta: SplitMetadata,
                         run_dir: Path, split: str) -> None:
    """Métricas por fonte, por origem e por faixa de comprimento (detecta atalhos residuais)."""
    for name, groups in (("source", meta.source), ("origin", meta.origin), ("bin", meta.bin)):
        write_json(run_dir / f"metrics_by_{name}_{split}.json", _grouped_metrics(y_true, y_pred, y_score, groups))


def save_error_samples(meta: SplitMetadata, y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray,
                       run_dir: Path, split: str, limit: int = 100) -> None:
    """Erros mais confiantes de cada tipo, que são o material qualitativo do TCC."""
    wrong = np.nonzero(y_true != y_pred)[0]
    fp = sorted((i for i in wrong if y_true[i] == 0), key=lambda i: -y_score[i])
    fn = sorted((i for i in wrong if y_true[i] == 1), key=lambda i: y_score[i])

    def row(i: int) -> dict[str, Any]:
        return {"text": meta.text[i], "score_dd": round(float(y_score[i]), 6), "source": meta.source[i],
                "bin": meta.bin[i]}

    write_json(run_dir / f"errors_{split}.json", {
        "falsos_positivos": [row(i) for i in fp[:limit]], "falsos_negativos": [row(i) for i in fn[:limit]],
        "total_falsos_positivos": len(fp), "total_falsos_negativos": len(fn)})


def package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("torch", "transformers", "accelerate", "datasets", "tokenizers", "scikit-learn", "numpy"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "ausente"
    return versions


def save_run_manifest(payload: dict[str, Any], run_dir: Path) -> None:
    """Grava o manifesto. Um `codigo` vindo no payload (capturado no início do processo) tem precedência."""
    payload = {"codigo": code_version(), **payload, "packages": package_versions(),
               "platform": {"system": platform.system(), "release": platform.release(),
                            "machine": platform.machine(), "mac_ver": platform.mac_ver()[0]}}
    write_json(run_dir / "run_manifest.json", payload)


def comparison_tables(runs: dict[str, dict[str, float]]) -> tuple[str, str]:
    """Tabela comparativa no teste, em Markdown e em LaTeX."""
    header = ["modelo", *REPORT_METRICS]
    rows = [[name, *[f"{m.get(k, float('nan')):.4f}" for k in REPORT_METRICS]] for name, m in runs.items()]
    markdown = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    markdown += ["| " + " | ".join(r) + " |" for r in rows]
    latex = ["\\begin{tabular}{l" + "r" * len(REPORT_METRICS) + "}", "\\hline",
             " & ".join(h.replace("_", "\\_") for h in header) + " \\\\", "\\hline",
             *[" & ".join(r).replace("_", "\\_") + " \\\\" for r in rows], "\\hline", "\\end{tabular}"]
    return "\n".join(markdown) + "\n", "\n".join(latex) + "\n"
