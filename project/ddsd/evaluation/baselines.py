"""Baselines clássicos: TF-IDF (n-gramas 1-3) + LinearSVC calibrado e + Regressão Logística.

- vocabulário limitado a 50 mil termos, `min_df=2`;
- `C ∈ {0,1; 1; 10}` por GridSearchCV (5-fold estratificado, F1 macro), só no treino;
- LinearSVC em `CalibratedClassifierCV` (Platt) para ter probabilidade.

Mesma disciplina dos Transformers: limiar escolhido na validação e teste avaliado
uma única vez. Os artefatos têm o mesmo layout de `models/<execução>/`, para que
o bootstrap e as tabelas tratem todos os modelos da mesma forma.

O modelo ajustado (vetorizador, classificador e limiar) fica em `modelo.joblib`.
`BaselineClassifier` o carrega com a mesma interface do `DddClassifier`, e assim
os baselines entram nas avaliações externas (texto real e ASR) como os
Transformers. O `joblib` usa pickle, então só se carregam artefatos produzidos
por este projeto, conferidos pelo SHA-256 gravado no manifesto do treino ou, no
modelo publicado no Hugging Face, no `.json` ao lado do `.joblib`.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC

from ddsd.classifiers.model import ArtifactError, Prediction
from ddsd.common.io import file_sha256, read_json, read_jsonl, write_json
from ddsd.observability.logger import get_logger
from ddsd.training import reporting
from ddsd.training.config import LABEL2ID
from ddsd.training.data import SplitMetadata
from ddsd.training.metrics import apply_threshold, best_threshold, classification_metrics

log = get_logger("baselines")

C_GRID = (0.1, 1.0, 10.0)
SEED = 42
MODEL_FILE = "modelo.joblib"


def _load(path: Path) -> tuple[list[str], np.ndarray, SplitMetadata]:
    rows = list(read_jsonl(path))
    meta = SplitMetadata(text=[r["text"] for r in rows], source=[r["source"] for r in rows],
                         origin=[r.get("origin", "") for r in rows], bin=[r.get("bin", "") for r in rows],
                         words=[r.get("words", 0) for r in rows])
    return meta.text, np.array([LABEL2ID[r["label"]] for r in rows]), meta


def train_baseline(kind: str, data_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Treina, escolhe o limiar na validação, avalia o teste uma vez e grava os artefatos."""
    started = time.perf_counter()
    run_dir = output_dir / f"baseline_{kind}"
    run_dir.mkdir(parents=True, exist_ok=True)
    x_train, y_train, _ = _load(data_dir / "train.jsonl")
    x_val, y_val, _ = _load(data_dir / "val.jsonl")
    x_test, y_test, test_meta = _load(data_dir / "test.jsonl")
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=50_000, min_df=2, sublinear_tf=True,
                                 token_pattern=r"(?u)\b\w+\b", lowercase=False)
    train_matrix = vectorizer.fit_transform(x_train)
    if kind == "svm":
        estimator: Any = LinearSVC(dual=False, max_iter=5000, random_state=SEED)
    elif kind == "logreg":
        estimator = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=SEED)
    else:
        raise ValueError(kind)
    search = GridSearchCV(estimator, {"C": list(C_GRID)}, scoring="f1_macro", n_jobs=-1, refit=True,
                          cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED))
    search.fit(train_matrix, y_train)
    best_c = float(search.best_params_["C"])
    model = search.best_estimator_
    if kind == "svm":
        model = CalibratedClassifierCV(LinearSVC(dual=False, max_iter=5000, random_state=SEED, C=best_c),
                                       cv=5, method="sigmoid")
        model.fit(train_matrix, y_train)
    val_score = model.predict_proba(vectorizer.transform(x_val))[:, 1]
    val_metrics = classification_metrics(y_val, (val_score >= 0.5).astype(int), val_score)
    threshold, _ = best_threshold(y_val, val_score)
    joblib.dump({"kind": kind, "vectorizer": vectorizer, "model": model, "threshold": threshold},
                run_dir / MODEL_FILE, compress=3)
    test_score = model.predict_proba(vectorizer.transform(x_test))[:, 1]
    test_pred = apply_threshold(test_score, threshold)
    test_metrics = classification_metrics(y_test, test_pred, test_score)
    write_json(run_dir / "metrics_val.json", val_metrics)
    write_json(run_dir / "metrics_test.json", test_metrics)
    reporting.save_predictions(y_test, test_pred, test_score, test_meta, run_dir, "test")
    reporting.save_grouped_metrics(y_test, test_pred, test_score, test_meta, run_dir, "test")
    reporting.save_error_samples(test_meta, y_test, test_pred, test_score, run_dir, "test")
    result = {"model_name": f"baseline_{kind}", "decision_threshold": threshold, "best_c": best_c,
              "cv_f1_macro": float(search.best_score_), "test_metrics": test_metrics, "val_metrics": val_metrics,
              "wall_time_s": round(time.perf_counter() - started, 1), "vocabulary": len(vectorizer.vocabulary_),
              "model_file": MODEL_FILE, "model_sha256": file_sha256(run_dir / MODEL_FILE)}
    reporting.save_run_manifest({"result": result, "data_dir": str(data_dir)}, run_dir)
    log.info("baseline %s: C=%g limiar=%.3f | TESTE acc %.4f f1_macro %.4f EER %.4f (%.0f s)", kind, best_c,
             threshold, test_metrics["accuracy"], test_metrics["f1_macro"], test_metrics["eer"],
             result["wall_time_s"])
    return result


class BaselineClassifier:
    """Baseline TF-IDF treinado, com a interface do `DddClassifier` (texto já normalizado).

    `run` é o modelo publicado no Hugging Face (`models/ddsd-ptbr-tfidf/<modelo>.joblib`, com o
    `.json` ao lado) ou a execução de treino (`models/baseline_<tipo>/`, de `ddsd eval baselines`).
    """

    def __init__(self, run: Path) -> None:
        if run.suffix == ".joblib":
            model_path, info_path = run, run.with_suffix(".json")
            if not (model_path.exists() and info_path.exists()):
                raise ArtifactError(f"{run} ou {info_path.name} não existe; rode `uv run ddsd download-models`")
            info = read_json(info_path)
            expected_sha, expected_threshold = info.get("sha256"), info.get("limiar_de_decisao")
        else:
            model_path = run / MODEL_FILE
            if not model_path.exists():
                raise ArtifactError(f"{model_path} não existe; rode `ddsd eval baselines` de novo para salvar o modelo")
            result = read_json(run / "run_manifest.json")["result"]
            expected_sha, expected_threshold = result.get("model_sha256"), result.get("decision_threshold")
        if file_sha256(model_path) != expected_sha:
            raise ArtifactError(f"{model_path}: SHA-256 não confere com o registrado (artefato alterado?)")
        artifact = joblib.load(model_path)
        self.name = f"baseline_{artifact['kind']}"
        self.run_dir = run
        self.vectorizer = artifact["vectorizer"]
        self.model = artifact["model"]
        self.threshold = float(artifact["threshold"])
        if expected_threshold is None or abs(self.threshold - float(expected_threshold)) > 1e-12:
            raise ArtifactError(f"{run}: limiar do modelo difere do registrado")
        log.info("%s carregado de %s (limiar %.3f)", self.name, run.name, self.threshold)

    def _predict(self, texts: list[str]) -> list[Prediction]:
        scores = self.model.predict_proba(self.vectorizer.transform(texts))[:, 1]
        return [Prediction(self.name, float(p), self.threshold, "DD" if p >= self.threshold else "NDD",
                           0.0, len(t.split()), False) for t, p in zip(texts, scores, strict=True)]

    def predict(self, normalized_text: str) -> Prediction:
        started = time.perf_counter()
        prediction = self._predict([normalized_text])[0]
        return dataclasses.replace(prediction, latency_ms=(time.perf_counter() - started) * 1000)

    def predict_batch(self, normalized_texts: list[str], batch_size: int = 4096) -> list[Prediction]:
        out: list[Prediction] = []
        for start in range(0, len(normalized_texts), batch_size):
            out += self._predict(normalized_texts[start : start + batch_size])
        return out

    def warm_up(self) -> None:
        self.predict("abre o navegador")
