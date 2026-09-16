"""
Descrição:
    Núcleo de uma execução de baseline: vetoriza, busca hiperparâmetro por
    CV, calibra o limiar de decisão na validação e avalia o teste uma única
    vez — a mesma disciplina de `src.training.engine`, para que baselines e
    Transformers sejam comparáveis sem ressalvas metodológicas.

    Ordem fixa:

    ```
    carrega splits -> ajusta TF-IDF no treino -> busca C por CV (treino)
    -> treina o estimador final no treino completo
    -> avalia VALIDAÇÃO -> escolhe limiar na VALIDAÇÃO
    -> avalia TESTE uma única vez, com o limiar congelado -> grava artefatos
    ```

Uso:
    from src.baseline.engine import run
    result = run("svm", BaselineConfig())
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from src.baseline.config import BaselineConfig
from src.baseline.data import load_split
from src.baseline.models import build_final_estimator, build_search_estimator
from src.baseline.tuning import grid_search
from src.baseline.vectorize import build_vectorizer
from src.data_prep import console
from src.training import reporting
from src.training.config import POSITIVE_ID
from src.training.metrics import apply_threshold, best_threshold, classification_metrics


@dataclass(slots=True)
class RunResult:
    """
    Descrição:
        Resultado de uma execução de baseline.

    Atributos:
        model_key: `"svm"` ou `"logreg"`.
        best_params: Hiperparâmetros escolhidos pela busca.
        threshold: Limiar de decisão calibrado na validação.
        val_metrics: Métricas na validação, com o limiar já aplicado.
        test_metrics: Métricas no teste, avaliado uma única vez.
        run_dir: Diretório onde os artefatos foram gravados.
    """

    model_key: str
    best_params: dict[str, Any]
    threshold: float
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    run_dir: str


def run(model_key: str, config: BaselineConfig) -> RunResult:
    """
    Descrição:
        Executa um baseline do início ao fim e grava os artefatos.

    Args:
        model_key: `"svm"` ou `"logreg"`.
        config: Configuração da execução.

    Retorna:
        `RunResult` com métricas e caminho de saída.
    """
    console.header(f"baseline: {model_key}")
    start = time.time()

    train = load_split(
        config.data_dir / "train.jsonl",
        max_samples=config.max_train_samples,
        seed=config.seed,
    )
    val = load_split(config.data_dir / "val.jsonl")
    test = load_split(config.data_dir / "test.jsonl")
    console.info(f"train={len(train)} val={len(val)} test={len(test)}")

    vectorizer = build_vectorizer(config)
    X_train = vectorizer.fit_transform(train.texts)
    X_val = vectorizer.transform(val.texts)
    X_test = vectorizer.transform(test.texts)
    console.info(f"vocabulário TF-IDF: {len(vectorizer.vocabulary_)} termos")

    search_estimator = build_search_estimator(model_key, config)
    search = grid_search(
        search_estimator,
        {"C": list(config.c_grid)},
        X_train,
        train.labels,
        cv=config.cv,
        seed=config.seed,
    )
    console.info(f"melhor C={search.best_params_['C']} (f1_macro CV={search.best_score_:.4f})")

    final_estimator = build_final_estimator(model_key, config, search.best_params_)
    final_estimator.fit(X_train, train.labels)

    val_scores = final_estimator.predict_proba(X_val)[:, POSITIVE_ID]
    threshold, threshold_value = best_threshold(val.labels, val_scores, metric="f1_macro")
    val_preds = apply_threshold(val_scores, threshold)
    val_metrics = classification_metrics(val.labels, val_preds, val_scores)
    console.info(f"limiar calibrado na validação: {threshold:.4f} (f1_macro={threshold_value:.4f})")

    test_scores = final_estimator.predict_proba(X_test)[:, POSITIVE_ID]
    test_preds = apply_threshold(test_scores, threshold)
    test_metrics = classification_metrics(test.labels, test_preds, test_scores)

    run_dir = config.run_dir(model_key)
    reporting.save_metrics(val_metrics, run_dir, "val")
    reporting.save_metrics(test_metrics, run_dir, "test")
    reporting.save_confusion_matrix(test.labels, test_preds, run_dir, "test")
    reporting.save_curves(test.labels, test_scores, run_dir, "test", test_metrics["eer"])
    reporting.save_predictions(test.labels, test_preds, test_scores, test.sources, run_dir, "test")
    reporting.save_per_source_metrics(
        test.labels, test_preds, test_scores, test.sources, run_dir, "test"
    )
    reporting.save_error_samples(test.texts, test.labels, test_preds, test_scores, run_dir, "test")
    reporting.save_run_manifest(
        {
            "model_key": model_key,
            "best_params": search.best_params_,
            "cv_best_score_f1_macro": float(search.best_score_),
            "threshold": threshold,
            "config": {
                "ngram_range": list(config.ngram_range),
                "max_features": config.max_features,
                "min_df": config.min_df,
                "c_grid": list(config.c_grid),
                "cv": config.cv,
                "calibration_cv": config.calibration_cv,
                "seed": config.seed,
            },
            "vocab_size": len(vectorizer.vocabulary_),
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "elapsed_seconds": round(time.time() - start, 1),
        },
        run_dir,
    )
    reporting.log_metrics(test_metrics, f"baseline_{model_key} — teste")

    return RunResult(
        model_key=model_key,
        best_params=search.best_params_,
        threshold=threshold,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        run_dir=str(run_dir),
    )
