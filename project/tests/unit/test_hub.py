"""Classificadores do Hugging Face: cópia local, download sob demanda e as duas origens dos artefatos."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from test_evaluation import _tiny_corpus

from ddsd.classifiers import hub
from ddsd.classifiers.model import ArtifactError, has_classifier, locate
from ddsd.common.io import file_sha256, read_json
from ddsd.realtime.config import PIPELINE_CHOICES, load_pipeline_config

REPOS = ("dono/ddsd-ptbr-bertimbau", "dono/ddsd-ptbr-tfidf")


def test_repo_for_maps_paths_inside_the_local_copy(tmp_path: Path) -> None:
    assert hub.local_dir(REPOS[0], tmp_path) == tmp_path / "ddsd-ptbr-bertimbau"
    assert hub.repo_for(tmp_path / "ddsd-ptbr-bertimbau", REPOS, tmp_path) == REPOS[0]
    assert hub.repo_for(tmp_path / "ddsd-ptbr-tfidf" / "svm.joblib", REPOS, tmp_path) == REPOS[1]
    assert hub.repo_for(tmp_path / "bertimbau-s44", REPOS, tmp_path) is None  # execução de treino: só local


def test_repo_for_follows_a_symlinked_models_dir(tmp_path: Path) -> None:
    real = tmp_path / "outro_disco"
    real.mkdir()
    link = tmp_path / "models"
    link.symlink_to(real, target_is_directory=True)
    configured = (link / "ddsd-ptbr-tfidf" / "svm.joblib").resolve()  # como a configuração entrega o caminho
    assert configured.is_relative_to(real.resolve())
    assert hub.repo_for(configured, REPOS, link) == REPOS[1]


def test_ensure_downloads_only_what_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_download(repo_id: str, models_dir: Path) -> Path:
        calls.append(repo_id)
        target = hub.local_dir(repo_id, models_dir)
        target.mkdir(parents=True)
        (target / "svm.joblib").write_bytes(b"x")
        return target

    monkeypatch.setattr(hub, "download", fake_download)
    wanted = tmp_path / "ddsd-ptbr-tfidf" / "svm.joblib"
    assert hub.ensure(wanted, REPOS, tmp_path) == wanted and calls == [REPOS[1]]
    hub.ensure(wanted, REPOS, tmp_path)  # já no disco: não baixa de novo
    hub.ensure(tmp_path / "bertimbau-s44", REPOS, tmp_path)  # fora dos repositórios: nada a baixar
    assert calls == [REPOS[1]]


def test_locate_reads_the_threshold_of_both_origins(tmp_path: Path) -> None:
    published = tmp_path / "ddsd-ptbr-bertimbau"
    published.mkdir()
    (published / "config.json").write_text(json.dumps({"decision_threshold": 0.685}))
    (published / "treinamento.json").write_text(json.dumps({"limiar_de_decisao": 0.685}))
    assert has_classifier(published) and locate("bertimbau", published) == (0.685, published)
    (published / "treinamento.json").write_text(json.dumps({"limiar_de_decisao": 0.5}))
    with pytest.raises(ArtifactError, match="treinamento.json"):
        locate("bertimbau", published)

    run = tmp_path / "bertimbau-s44"
    (run / "checkpoints" / "checkpoint-8").mkdir(parents=True)
    (run / "checkpoints" / "checkpoint-8" / "config.json").write_text("{}")
    (run / "run_manifest.json").write_text(json.dumps({"result": {"decision_threshold": 0.4,
                                                                  "best_checkpoint": "checkpoints/checkpoint-8"}}))
    assert locate("bertimbau", run) == (0.4, run / "checkpoints" / "checkpoint-8")

    empty = tmp_path / "vazio"
    empty.mkdir()
    assert not has_classifier(empty)
    with pytest.raises(ArtifactError, match="download-models"):
        locate("bertimbau", empty)


def test_baseline_loads_the_published_joblib_like_the_training_run(tmp_path: Path) -> None:
    from ddsd.evaluation.baselines import MODEL_FILE, BaselineClassifier, train_baseline

    _tiny_corpus(tmp_path / "splits")
    train_baseline("logreg", tmp_path / "splits", tmp_path / "models")
    run_dir = tmp_path / "models" / "baseline_logreg"
    published = tmp_path / "ddsd-ptbr-tfidf" / "regressao_logistica.joblib"
    published.parent.mkdir()
    shutil.copy(run_dir / MODEL_FILE, published)
    result = read_json(run_dir / "run_manifest.json")["result"]
    info = {"sha256": file_sha256(published), "limiar_de_decisao": result["decision_threshold"]}
    published.with_suffix(".json").write_text(json.dumps(info))

    trained, from_hub = BaselineClassifier(run_dir), BaselineClassifier(published)
    assert from_hub.name == trained.name == "baseline_logreg" and from_hub.threshold == trained.threshold
    texts = [json.loads(line)["text"] for line in (run_dir / "predictions_test.jsonl").read_text().splitlines()]
    assert from_hub.predict_batch(texts) == trained.predict_batch(texts)

    published.with_suffix(".json").write_text(json.dumps({**info, "sha256": "0" * 64}))
    with pytest.raises(ArtifactError, match="SHA-256"):
        BaselineClassifier(published)


def test_default_config_takes_every_classifier_from_a_hub_repo() -> None:
    cfg = load_pipeline_config()
    for name in PIPELINE_CHOICES:
        assert hub.repo_for(cfg.classifiers.run_for(name), cfg.hub.repos) is not None, name
