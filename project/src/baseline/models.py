"""
Descrição:
    Estimadores dos dois baselines.

    O SVM usa `LinearSVC`, não `SVC(kernel="linear")`: com ~150 mil exemplos
    de treino por fold, `SVC` (que resolve a formulação dual completa) é
    inviável, enquanto `LinearSVC` (liblinear) escala linearmente. Como
    `LinearSVC` não expõe `predict_proba`, o modelo final é envolvido em
    `CalibratedClassifierCV` (Platt scaling) para produzir probabilidades
    compatíveis com `src.training.metrics.best_threshold` e
    `classification_metrics`, que esperam um score contínuo em `[0, 1]`.

    A Regressão Logística já expõe `predict_proba` nativamente e não precisa
    de calibração adicional.

Uso:
    from src.baseline.models import build_search_estimator, build_final_estimator
"""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.baseline.config import BaselineConfig


def build_search_estimator(model_key: str, config: BaselineConfig) -> BaseEstimator:
    """
    Descrição:
        Constrói o estimador usado dentro do `GridSearchCV` para escolher `C`.

        Para o SVM, a busca roda sobre o `LinearSVC` puro (sem calibração):
        calibrar a cada candidato de `C` multiplicaria o custo por
        `calibration_cv` sem melhorar a escolha de `C`, que depende só da
        fronteira de decisão.

    Args:
        model_key: `"svm"` ou `"logreg"`.
        config: Configuração do baseline.

    Retorna:
        Estimador não ajustado.

    Levanta:
        ValueError: Se `model_key` não for reconhecida.
    """
    if model_key == "svm":
        return LinearSVC(dual=False, max_iter=config.svm_max_iter, random_state=config.seed)
    if model_key == "logreg":
        return LogisticRegression(
            solver="lbfgs", max_iter=config.logreg_max_iter, random_state=config.seed
        )
    raise ValueError(f"baseline desconhecido: {model_key!r}")


def build_final_estimator(
    model_key: str, config: BaselineConfig, best_params: dict[str, Any]
) -> BaseEstimator:
    """
    Descrição:
        Constrói o estimador final, treinado no conjunto de treino completo
        com o `C` escolhido pela busca.

        Para o SVM, envolve o `LinearSVC` calibrado em `CalibratedClassifierCV`
        para obter `predict_proba`. Para a Regressão Logística, é o próprio
        estimador (já teria sido refeito pelo `GridSearchCV`, mas construir de
        novo aqui deixa o fluxo dos dois baselines simétrico em `engine.py`).

    Args:
        model_key: `"svm"` ou `"logreg"`.
        config: Configuração do baseline.
        best_params: Melhores hiperparâmetros encontrados pela busca
            (`{"C": valor}`).

    Retorna:
        Estimador não ajustado, pronto para `fit` no treino completo.

    Levanta:
        ValueError: Se `model_key` não for reconhecida.
    """
    if model_key == "svm":
        base = LinearSVC(
            dual=False,
            max_iter=config.svm_max_iter,
            random_state=config.seed,
            **best_params,
        )
        return CalibratedClassifierCV(base, method="sigmoid", cv=config.calibration_cv)
    if model_key == "logreg":
        return LogisticRegression(
            solver="lbfgs",
            max_iter=config.logreg_max_iter,
            random_state=config.seed,
            **best_params,
        )
    raise ValueError(f"baseline desconhecido: {model_key!r}")
