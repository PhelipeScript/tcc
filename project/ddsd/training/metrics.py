"""Métricas da tarefa (DD positivo), EER e escolha do limiar na validação.

O EER interpola a curva ROC e resolve fpr(x) = fnr(x) com Brent. A implementação
é conferida contra o caso analítico de duas normais com d' = 2, cujo EER =
Φ(-1) ≈ 0,1587 (`tests/unit/test_training_metrics.py`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from ddsd.training.config import POSITIVE_ID

METRIC_KEYS: tuple[str, ...] = (
    "accuracy", "balanced_accuracy", "precision_dd", "recall_dd", "f1_dd", "precision_macro",
    "recall_macro", "f1_macro", "roc_auc", "pr_auc", "eer", "eer_threshold", "mcc",
)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax numericamente estável por linha."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


def positive_scores(logits: np.ndarray) -> np.ndarray:
    """P(DD) a partir dos logits `(n, 2)`."""
    return softmax(np.asarray(logits, dtype=np.float64))[:, POSITIVE_ID]


def equal_error_rate(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """(EER, limiar no EER), ou (nan, nan) se `y_true` tiver uma classe só."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    try:
        eer = float(brentq(lambda x: interp1d(fpr, fnr)(x) - x, 0.0, 1.0))
        threshold = float(interp1d(fpr, thresholds)(eer))
    except (ValueError, RuntimeError):
        # Curva degenerada (ex.: classificador perfeito): vértice mais próximo de fpr == fnr.
        index = int(np.nanargmin(np.abs(fnr - fpr)))
        eer = float((fpr[index] + fnr[index]) / 2.0)
        threshold = float(thresholds[index])
    return eer, threshold


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Conjunto completo de métricas (binárias em DD e macro)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    y_score = np.asarray(y_score, dtype=np.float64)
    precision_dd, recall_dd, f1_dd, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=POSITIVE_ID, zero_division=0)
    precision_macro, recall_macro, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    eer, eer_threshold = equal_error_rate(y_true, y_score)
    both = len(np.unique(y_true)) >= 2
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if both else float("nan"),
        "precision_dd": float(precision_dd),
        "recall_dd": float(recall_dd),
        "f1_dd": float(f1_dd),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_score)) if both else float("nan"),
        "eer": eer,
        "eer_threshold": eer_threshold,
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def build_compute_metrics() -> Callable[[Any], dict[str, float]]:
    """`compute_metrics` do `Trainer`: logits → probabilidades → métricas (predição por argmax)."""

    def compute_metrics(eval_prediction: Any) -> dict[str, float]:
        logits = eval_prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(eval_prediction.label_ids)
        return classification_metrics(labels, logits.argmax(axis=-1), softmax(logits)[:, POSITIVE_ID])

    return compute_metrics


def best_threshold(y_true: np.ndarray, y_score: np.ndarray, metric: str = "f1_macro",
                   steps: int = 201) -> tuple[float, float]:
    """Limiar que maximiza `metric`. Use **somente na validação**; no teste ele entra congelado."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    best_value, best = -1.0, 0.5
    for threshold in np.linspace(0.0, 1.0, steps)[1:-1]:
        predictions = (y_score >= threshold).astype(int)
        if metric == "f1_macro":
            value = f1_score(y_true, predictions, average="macro", zero_division=0)
        else:
            value = classification_metrics(y_true, predictions, y_score)[metric]
        if value > best_value:
            best_value, best = float(value), float(threshold)
    return best, best_value


def apply_threshold(y_score: np.ndarray, threshold: float) -> np.ndarray:
    """Binariza scores com um limiar."""
    return (np.asarray(y_score, dtype=np.float64) >= threshold).astype(int)


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Matriz 2×2 com ordem fixa (NDD, DD): linhas = verdadeiro, colunas = predito."""
    return confusion_matrix(y_true, y_pred, labels=[0, 1])
