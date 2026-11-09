"""Métricas de treino: EER no caso analítico, escolha de limiar e orientação da matriz de confusão."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from ddsd.training.metrics import (apply_threshold, best_threshold, classification_metrics, confusion,
                                   equal_error_rate, positive_scores)


def test_eer_matches_analytic_gaussian_case() -> None:
    """Duas normais unitárias separadas por d' = 2: EER teórico = Φ(-1) ≈ 0,1587."""
    rng = np.random.default_rng(42)
    n = 200_000
    scores = np.r_[rng.normal(0, 1, n), rng.normal(2, 1, n)]
    labels = np.r_[np.zeros(n), np.ones(n)]
    eer, threshold = equal_error_rate(labels, scores)
    assert eer == pytest.approx(norm.cdf(-1), abs=0.003)
    assert threshold == pytest.approx(1.0, abs=0.05)  # ponto médio entre as médias


def test_eer_single_class_is_nan() -> None:
    eer, threshold = equal_error_rate(np.ones(10), np.linspace(0, 1, 10))
    assert np.isnan(eer) and np.isnan(threshold)


def test_positive_scores_softmax_column_dd() -> None:
    logits = np.array([[0.0, 0.0], [0.0, 10.0], [10.0, 0.0]])
    scores = positive_scores(logits)
    assert scores == pytest.approx([0.5, 1.0, 0.0], abs=1e-4)


def test_best_threshold_finds_separating_point() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    threshold, value = best_threshold(y_true, y_score)
    assert 0.3 < threshold <= 0.7 and value == 1.0
    assert (apply_threshold(y_score, threshold) == y_true).all()


def test_confusion_orientation_is_fixed() -> None:
    matrix = confusion(np.array([1, 1, 1]), np.array([1, 1, 0]))
    assert matrix.tolist() == [[0, 0], [1, 2]]  # linhas = verdadeiro (NDD, DD)


def test_classification_metrics_dd_is_positive() -> None:
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])
    m = classification_metrics(y_true, y_pred, np.array([0.9, 0.4, 0.2, 0.1]))
    assert m["precision_dd"] == 1.0 and m["recall_dd"] == 0.5


def test_parse_pmset_power_conditions() -> None:
    from ddsd.common.system import parse_pmset

    battery = ("Now drawing from 'Battery Power'\n -InternalBattery-0 (id=24117347)\t20%; discharging; "
               "0:41 remaining present: true\n")
    assert parse_pmset(battery, " lowpowermode         0\n") == {
        "fonte_energia": "bateria", "bateria_pct": 20, "modo_pouca_energia": False}
    ac = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t85%; charging; 0:30 remaining present: true\n"
    assert parse_pmset(ac, " lowpowermode         1\n")["fonte_energia"] == "tomada"
    assert parse_pmset(ac, " lowpowermode         1\n")["modo_pouca_energia"] is True
    assert parse_pmset("", "") == {"fonte_energia": "desconhecida", "bateria_pct": None, "modo_pouca_energia": None}


def test_code_version_tracks_commit_and_code_changes(tmp_path) -> None:
    import subprocess

    from ddsd.common.system import code_version

    assert code_version(tmp_path)["commit"] is None  # sem repositório: registra o motivo, não quebra

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                       check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "m.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "paper").mkdir()
    (tmp_path / "paper" / "cap3.tex").write_text("texto\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-q", "-m", "inicial")
    clean = code_version(tmp_path)
    assert len(clean["commit"]) == 40 and clean["alteracoes_nao_commitadas"] is False

    (tmp_path / "paper" / "cap3.tex").write_text("texto revisado\n", encoding="utf-8")
    assert code_version(tmp_path)["alteracoes_nao_commitadas"] is False  # paper/ não é código
    (tmp_path / "src" / "m.py").write_text("x = 2\n", encoding="utf-8")
    (tmp_path / "src" / "novo.py").write_text("y = 1\n", encoding="utf-8")
    dirty = code_version(tmp_path)
    assert dirty["alteracoes_nao_commitadas"] is True
    assert sorted(dirty["arquivos_alterados"]) == ["src/m.py", "src/novo.py"]


def test_manifest_keeps_code_version_captured_at_start(tmp_path, monkeypatch) -> None:
    import json

    from ddsd.training import engine, reporting

    monkeypatch.setattr(reporting, "code_version", lambda: {"commit": "novo", "arquivos_alterados": []})
    reporting.save_run_manifest({"codigo": {"commit": "carregado"}}, tmp_path)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["codigo"] == {"commit": "carregado"}  # o do início do processo vence
    reporting.save_run_manifest({}, tmp_path)
    assert json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))["codigo"]["commit"] == "novo"

    monkeypatch.setattr(engine, "CODE_AT_IMPORT", {"commit": "a", "arquivos_alterados": []})
    monkeypatch.setattr(engine, "code_version", lambda: {"commit": "a", "arquivos_alterados": []})
    assert engine._code_drift() == {}
    monkeypatch.setattr(engine, "code_version", lambda: {"commit": "b", "arquivos_alterados": []})
    assert engine._code_drift() == {"codigo_no_repositorio_ao_fim": {"commit": "b", "arquivos_alterados": []}}
