"""
Descrição:
    Métricas de avaliação da tarefa DDSD.

    Calcula o conjunto exigido pela metodologia do TCC: acurácia, precisão,
    recall, F1, ROC-AUC e **EER** (equal error rate), além de MCC e acurácia
    balanceada como leituras complementares.

    O EER é o ponto da curva ROC em que a taxa de falsos positivos iguala a de
    falsos negativos. Aqui ele é obtido por **interpolação** entre os vértices
    da curva, não pelo vértice mais próximo: com ~18 mil pontos de teste, o
    vértice mais próximo pode errar o valor em cerca de 0,1 ponto percentual, o
    que importa quando o critério de aceite do trabalho é "EER < 30%".

    A classe positiva é DD (índice 1). Assim, um falso positivo é o assistente
    acordar durante conversa de fundo — exatamente o erro crítico do sistema.

    Este módulo depende apenas de NumPy, SciPy e scikit-learn: não importa
    PyTorch nem transformers, então é testável sem GPU.

Uso:
    from src.training.metrics import build_compute_metrics, classification_metrics
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
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

from src.training.config import POSITIVE_ID

#: Chaves produzidas por `classification_metrics`, na ordem de exibição.
METRIC_KEYS: tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "precision_dd",
    "recall_dd",
    "f1_dd",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "roc_auc",
    "pr_auc",
    "eer",
    "eer_threshold",
    "mcc",
)


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Descrição:
        Converte logits em probabilidades, de forma numericamente estável
        (subtraindo o máximo antes de exponenciar).

    Args:
        logits: Matriz de forma `(n, n_classes)`.

    Retorna:
        Matriz de probabilidades de mesma forma, com linhas somando 1.
    """
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


def positive_scores(logits: np.ndarray) -> np.ndarray:
    """
    Descrição:
        Extrai a probabilidade da classe positiva (DD) a partir dos logits.

    Args:
        logits: Matriz de forma `(n, 2)`.

    Retorna:
        Vetor de forma `(n,)` com `P(DD)`.
    """
    return softmax(np.asarray(logits, dtype=np.float64))[:, POSITIVE_ID]


def equal_error_rate(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """
    Descrição:
        Calcula o EER e o limiar em que ele ocorre.

        Interpola a curva ROC e resolve `fpr(x) - (1 - tpr(x)) = 0` com o método
        de Brent. Se a resolução falhar (curva degenerada, por exemplo com um
        classificador perfeito), cai para o vértice da ROC mais próximo da
        igualdade — o que mantém a função utilizável em qualquer entrada.

    Args:
        y_true: Rótulos verdadeiros (0 = NDD, 1 = DD).
        y_score: Score contínuo da classe positiva.

    Retorna:
        Tupla `(eer, limiar)`. Ambos em `[0, 1]`, ou `(nan, nan)` se `y_true`
        contiver apenas uma classe — caso em que a ROC é indefinida.
    """
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        # Sem as duas classes a ROC é indefinida. Devolver NaN em vez de
        # levantar exceção evita derrubar um treino inteiro por causa de um
        # subconjunto de avaliação homogêneo (acontece com --max-eval-samples).
        return float("nan"), float("nan")

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr

    try:
        eer = float(brentq(lambda x: interp1d(fpr, fnr)(x) - x, 0.0, 1.0))
        threshold = float(interp1d(fpr, thresholds)(eer))
    except (ValueError, RuntimeError):
        # Curva degenerada: usa o vértice mais próximo de fpr == fnr.
        index = int(np.nanargmin(np.abs(fnr - fpr)))
        eer = float((fpr[index] + fnr[index]) / 2.0)
        threshold = float(thresholds[index])

    return eer, threshold


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray
) -> dict[str, float]:
    """
    Descrição:
        Calcula o conjunto completo de métricas da tarefa.

        As métricas com sufixo `_dd` são binárias sobre a classe positiva (DD),
        e são a leitura operacional do sistema. As com sufixo `_macro` são a
        média não ponderada entre as duas classes, e são a métrica de seleção do
        melhor checkpoint — insensível a eventual desbalanceamento residual.

    Args:
        y_true: Rótulos verdadeiros (0 = NDD, 1 = DD).
        y_pred: Rótulos preditos.
        y_score: Score contínuo da classe positiva, usado por ROC-AUC e EER.

    Retorna:
        Dicionário com as chaves de `METRIC_KEYS`.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score, dtype=np.float64)

    precision_dd, recall_dd, f1_dd, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=POSITIVE_ID, zero_division=0
    )
    precision_macro, recall_macro, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    eer, eer_threshold = equal_error_rate(y_true, y_score)

    # ROC-AUC e PR-AUC também exigem as duas classes presentes.
    both_classes = len(np.unique(y_true)) >= 2
    roc_auc = float(roc_auc_score(y_true, y_score)) if both_classes else float("nan")
    pr_auc = float(average_precision_score(y_true, y_score)) if both_classes else float("nan")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_dd": float(precision_dd),
        "recall_dd": float(recall_dd),
        "f1_dd": float(f1_dd),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def build_compute_metrics() -> Callable[[Any], dict[str, float]]:
    """
    Descrição:
        Constrói a função `compute_metrics` passada ao `Trainer`.

        O `Trainer` entrega logits crus; a função converte para probabilidades,
        deriva a predição por argmax e calcula todas as métricas.

    Retorna:
        Função que recebe um `EvalPrediction` e devolve o dicionário de métricas.
    """

    def compute_metrics(eval_prediction: Any) -> dict[str, float]:
        """
        Descrição:
            Calcula as métricas de uma avaliação do `Trainer`.

        Args:
            eval_prediction: Objeto com `predictions` (logits) e `label_ids`.

        Retorna:
            Dicionário de métricas.
        """
        logits = eval_prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(eval_prediction.label_ids)
        scores = softmax(logits)[:, POSITIVE_ID]
        predictions = logits.argmax(axis=-1)
        return classification_metrics(labels, predictions, scores)

    return compute_metrics


def best_threshold(
    y_true: np.ndarray, y_score: np.ndarray, metric: str = "f1_macro", steps: int = 201
) -> tuple[float, float]:
    """
    Descrição:
        Varre limiares de decisão e devolve o que maximiza a métrica escolhida.

        Deve ser usada **somente na validação**. O limiar encontrado é então
        aplicado congelado ao conjunto de teste — escolher o limiar no próprio
        teste invalidaria a reserva do conjunto.

    Args:
        y_true: Rótulos verdadeiros da validação.
        y_score: Score contínuo da classe positiva.
        metric: Nome da métrica a maximizar (chave de `classification_metrics`).
        steps: Número de limiares avaliados no intervalo `(0, 1)`.

    Retorna:
        Tupla `(limiar, valor_da_metrica)`.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)

    candidates = np.linspace(0.0, 1.0, steps)[1:-1]
    best_value = -1.0
    best = 0.5
    for threshold in candidates:
        predictions = (y_score >= threshold).astype(int)
        value = f1_score(y_true, predictions, average="macro", zero_division=0)
        if metric != "f1_macro":
            value = classification_metrics(y_true, predictions, y_score)[metric]
        if value > best_value:
            best_value = float(value)
            best = float(threshold)
    return best, best_value


def apply_threshold(y_score: np.ndarray, threshold: float) -> np.ndarray:
    """
    Descrição:
        Binariza scores contínuos com um limiar arbitrário.

    Args:
        y_score: Score contínuo da classe positiva.
        threshold: Limiar de decisão.

    Retorna:
        Vetor de predições (0 ou 1).
    """
    return (np.asarray(y_score, dtype=np.float64) >= threshold).astype(int)


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Descrição:
        Calcula a matriz de confusão 2x2 com ordem de rótulos fixa `(NDD, DD)`,
        para que a orientação da matriz não dependa dos dados presentes.

    Args:
        y_true: Rótulos verdadeiros.
        y_pred: Rótulos preditos.

    Retorna:
        Matriz de forma `(2, 2)`, linhas = verdadeiro, colunas = predito.
    """
    return confusion_matrix(y_true, y_pred, labels=[0, 1])
