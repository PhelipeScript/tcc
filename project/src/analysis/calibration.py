"""
Descrição:
    Diagrama de confiabilidade (reliability diagram) a partir de
    `predictions_test.jsonl` — ilustra diretamente o overfitting de
    calibração/confiança identificado no treino: se o modelo está
    superconfiante, os pontos ficam sistematicamente abaixo da diagonal na
    faixa de score alto (confiança maior que a taxa real de acerto).

Uso:
    from src.analysis.calibration import load_predictions, plot_reliability_diagram
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.training.config import LABEL2ID


def load_predictions(predictions_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Descrição:
        Lê `predictions_test.jsonl` e monta os vetores de rótulo verdadeiro
        (0/1) e score contínuo da classe DD.

    Args:
        predictions_path: Caminho de `predictions_<split>.jsonl`.

    Retorna:
        Tupla `(y_true, y_score)`.
    """
    labels: list[int] = []
    scores: list[float] = []
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            labels.append(LABEL2ID[row["label"]])
            scores.append(row["score_dd"])
    return np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)


def plot_reliability_diagram(
    per_model: dict[str, tuple[np.ndarray, np.ndarray]], output_path: Path, n_bins: int = 15
) -> Path:
    """
    Descrição:
        Plota o diagrama de confiabilidade de um ou mais modelos sobrepostos,
        usando `sklearn.calibration.calibration_curve` com bins de largura
        igual sobre o score da classe DD.

    Args:
        per_model: Mapeamento `nome de exibição -> (y_true, y_score)`.
        output_path: Caminho do PNG a gravar.
        n_bins: Número de bins de score.

    Retorna:
        `output_path`.

    Levanta:
        ValueError: Se `per_model` estiver vazio.
    """
    if not per_model:
        raise ValueError("per_model vazio: nada para plotar")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    figure, axis = plt.subplots(figsize=(5.0, 5.0))
    axis.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8, label="calibração perfeita")
    for name, (y_true, y_score) in per_model.items():
        fraction_positive, mean_predicted = calibration_curve(
            y_true, y_score, n_bins=n_bins, strategy="uniform"
        )
        axis.plot(mean_predicted, fraction_positive, marker="o", label=name)

    axis.set_xlabel("confiança média predita (P(DD))")
    axis.set_ylabel("fração real de DD no bin")
    axis.set_title("Diagrama de confiabilidade — conjunto de teste")
    axis.legend(loc="lower right", fontsize="small")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path
