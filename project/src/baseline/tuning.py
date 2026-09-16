"""
Descrição:
    Busca de hiperparâmetro por validação cruzada, conforme o Experimento 1
    da metodologia do TCC ("validação cruzada 5-fold").

Uso:
    from src.baseline.tuning import grid_search
    search = grid_search(estimator, {"C": [0.1, 1.0, 10.0]}, X_train, y_train, cv=5)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def grid_search(
    estimator: BaseEstimator,
    param_grid: dict[str, Any],
    X: sp.spmatrix,
    y: np.ndarray,
    *,
    cv: int,
    seed: int,
    scoring: str = "f1_macro",
    n_jobs: int = -1,
) -> GridSearchCV:
    """
    Descrição:
        Roda `GridSearchCV` com validação cruzada estratificada.

        `scoring="f1_macro"` alinha o critério de seleção de hiperparâmetro
        com a métrica de seleção de checkpoint usada no treino dos
        Transformers (`metric_for_best_model` em
        `src.training.config.TrainingConfig`), tornando a comparação entre
        baselines e modelos justa.

    Args:
        estimator: Estimador não ajustado (`LinearSVC` ou `LogisticRegression`).
        param_grid: Grade de hiperparâmetros, ex. `{"C": (0.1, 1.0, 10.0)}`.
        X: Matriz de features do treino (TF-IDF, esparsa).
        y: Rótulos do treino.
        cv: Número de folds.
        seed: Semente do shuffle dos folds.
        scoring: Métrica otimizada pela busca.
        n_jobs: Paralelismo do `GridSearchCV` (`-1` usa todos os núcleos).

    Retorna:
        `GridSearchCV` já ajustado (`fit` foi chamado).
    """
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    search = GridSearchCV(
        estimator, param_grid, scoring=scoring, cv=splitter, n_jobs=n_jobs, refit=True
    )
    search.fit(X, y)
    return search
