"""Módulos de avaliação: bootstrap pareado, FP externo, resumos de ASR e de pipeline, WER."""

from __future__ import annotations

import numpy as np
import pytest

from ddsd.evaluation.asr_impact import summarize as asr_summarize
from ddsd.evaluation.asr_impact import wer
from ddsd.evaluation.bootstrap import check_alignment, paired_bootstrap
from ddsd.evaluation.external_text import false_positive_report
from ddsd.evaluation.pipeline_eval import summarize as pipeline_summarize


def preds(y_true, y_pred, score, text=None) -> dict[str, np.ndarray]:
    n = len(y_true)
    return {"text": np.array(text or [f"t{i}" for i in range(n)], dtype=object), "y_true": np.array(y_true),
            "y_pred": np.array(y_pred), "y_score": np.array(score, dtype=float)}


def test_bootstrap_detects_better_model_and_misalignment() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    good_score = np.clip(y * 0.6 + rng.normal(0.2, 0.15, 2000), 0, 1)
    bad_score = np.clip(y * 0.2 + rng.normal(0.4, 0.25, 2000), 0, 1)
    a = preds(y, (good_score >= 0.5).astype(int), good_score)
    b = preds(y, (bad_score >= 0.5).astype(int), bad_score)
    result = paired_bootstrap(a, b, n_bootstrap=200)
    assert result["accuracy"]["p_a_melhor"] > 0.99 and result["accuracy"]["ic95_inf"] > 0
    assert result["eer"]["p_a_melhor"] > 0.99 and result["eer"]["diferenca_media"] < 0
    shuffled = preds(y[::-1], b["y_pred"], b["y_score"])
    with pytest.raises(ValueError):
        check_alignment({"a": a, "b": shuffled})


def test_false_positive_report_by_corpus_and_bin() -> None:
    rows = [({"corpus": "coraa", "words": 2}, "DD", 0.9), ({"corpus": "coraa", "words": 2}, "NDD", 0.1),
            ({"corpus": "coraa", "words": 12}, "NDD", 0.2), ({"corpus": "nurcsp", "words": 3}, "DD", 0.8)]
    bins = [(2, 2, "2"), (3, 3, "3"), (4, 50, "4-50")]
    report = false_positive_report({"m": rows}, bins)
    assert report["m"]["coraa"]["fp_rate"] == pytest.approx(1 / 3)
    assert report["m"]["coraa"]["por_faixa"]["2"] == {"n": 2, "fp_rate": 0.5}
    assert report["m"]["nurcsp"]["fp_rate"] == 1.0


def test_false_positive_report_keeps_sentences_outside_the_corpus_bins() -> None:
    # Transcrição real com mais palavras que a última faixa do corpus (ex.: 64 > 50): entra na taxa do
    # corpus e aparece numa faixa própria, sem derrubar o relatório.
    rows = [({"corpus": "nurcsp", "words": 64}, "DD", 0.9), ({"corpus": "nurcsp", "words": 3}, "NDD", 0.1),
            ({"corpus": "nurcsp", "words": 1}, "NDD", 0.2)]
    report = false_positive_report({"m": rows}, [(2, 2, "2"), (3, 3, "3"), (4, 50, "4-50")])
    nurcsp = report["m"]["nurcsp"]
    assert nurcsp["n"] == 3 and nurcsp["fp_rate"] == pytest.approx(1 / 3)
    assert nurcsp["por_faixa"] == {"<2": {"n": 1, "fp_rate": 0.0}, "3": {"n": 1, "fp_rate": 0.0},
                                   ">50": {"n": 1, "fp_rate": 1.0}}
    assert list(nurcsp["por_faixa"]) == ["<2", "3", ">50"]


def test_wer_uses_normalization_on_both_sides() -> None:
    assert wer("Abre o navegador!", " abre o Navegador.") == 0.0
    assert wer("abre o navegador", "abre navegador") == pytest.approx(1 / 3)
    assert wer("", "") == 0.0 and wer("", "ruido") == 1.0


def test_wer_without_filled_pauses_removes_them_on_both_sides() -> None:
    from ddsd.evaluation.asr_impact import FILLED_PAUSES, wer_without_filled_pauses, without_filled_pauses

    assert without_filled_pauses("Eh, então... ahn, eu acho que sim") == "entao eu acho que sim"
    assert without_filled_pauses("Há dois anos, ah, foi lá") == "ha dois anos foi la"  # "há" fica
    assert "ha" not in FILLED_PAUSES
    assert wer_without_filled_pauses("eh eu fui la", "eu fui la") == 0.0
    assert wer("eh eu fui la", "eu fui la") == pytest.approx(1 / 4)
    assert wer_without_filled_pauses("ahn", "") == 0.0 and wer_without_filled_pauses("ahn", "casa") == 1.0


def test_asr_summary_rates_and_bands() -> None:
    records = [
        {"label": "DD", "wer": 0.0, "reliability": 0.9, "m:reference_norm": "DD", "m:hypothesis": "DD"},
        {"label": "DD", "wer": 0.5, "reliability": 0.3, "m:reference_norm": "DD", "m:hypothesis": "NDD"},
        {"label": "NDD", "wer": 0.0, "reliability": 0.8, "m:reference_norm": "NDD", "m:hypothesis": "DD"},
        {"label": "NDD", "wer": 0.05, "reliability": 0.7, "m:reference_norm": "NDD",
         "m:hypothesis": "NDD"},
    ]
    s = asr_summarize(records, ["m"])
    assert s["m"]["referencia"]["acuracia"] == 1.0
    assert s["m"]["asr"] == {"n": 4, "acuracia": 0.5, "fp_rate": 0.5, "fn_rate": 0.5}
    assert s["m"]["por_faixa_wer"][">0,3"]["n"] == 1
    assert s["spearman_confiabilidade_wer"]["rho"] < 0  # confiabilidade cai quando o WER sobe
    assert "asr_sem_referencia" not in s["m"]


def test_asr_summary_pairs_reference_and_asr_and_keeps_free_speech_apart() -> None:
    """A fala livre (sem roteiro) não entra na comparação referência × ASR, só no FP à parte."""
    paired = [
        {"label": "DD", "wer": 0.0, "reliability": 0.9, "m:reference_norm": "DD", "m:hypothesis": "DD"},
        {"label": "NDD", "wer": 0.2, "reliability": 0.6, "m:reference_norm": "NDD",
         "m:hypothesis": "NDD"},
    ]
    free = {"label": "NDD", "wer": None, "reliability": 0.5, "m:reference_norm": "NDD",
            "m:hypothesis": "DD"}
    s = asr_summarize([*paired, free], ["m"])
    assert s["n"] == 3 and s["n_com_referencia"] == 2
    assert s["m"]["referencia"]["n"] == s["m"]["asr"]["n"] == 2
    assert s["m"]["asr"]["acuracia"] == 1.0 and s["m"]["asr"]["fp_rate"] == 0.0
    free_rates = s["m"]["asr_sem_referencia"]
    assert free_rates["n"] == 1 and free_rates["fp_rate"] == 1.0 and np.isnan(free_rates["fn_rate"])


def test_pipeline_summary_activation_and_policy() -> None:
    records = [
        {"group": "g", "label": "DD", "utterances": 1, "activated": True, "activated_low_policy": True,
         "latencies_s": [0.8]},
        {"group": "g", "label": "NDD", "utterances": 1, "activated": True,
         "activated_low_policy": False, "latencies_s": [0.9]},
        {"group": "g", "label": "RUIDO", "utterances": 0, "activated": False,
         "activated_low_policy": False, "latencies_s": []},
    ]
    s = pipeline_summarize(records)["g"]
    assert s["DD"]["ativacao"] == 1.0 and s["NDD"]["ativacao"] == 1.0
    assert s["NDD"]["ativacao_politica_baixa_ndd"] == 0.0
    assert s["RUIDO"]["sem_fala_detectada"] == 1.0
    assert s["latencia_s"]["n"] == 2


def test_pipeline_summary_keeps_free_speech_apart_and_counts_fp_per_utterance() -> None:
    def decision(label: str, alert: bool = False) -> dict:
        return {"label": label, "reliability": {"alert": alert}}

    free = {"group": "g", "label": "NDD_LIVRE", "utterances": 4, "activated": True,
            "activated_low_policy": False, "latencies_s": [0.5] * 4, "audio_s": 60.0,
            "decisions": [decision("DD", alert=True), decision("NDD"), decision("NDD"), decision("NDD")]}
    sentence = {"group": "g", "label": "NDD", "utterances": 1, "activated": False,
                "activated_low_policy": False, "latencies_s": [0.4], "audio_s": 3.0, "decisions": [decision("NDD")]}
    s = pipeline_summarize([free, sentence])["g"]
    assert s["NDD"]["n"] == 1 and s["NDD"]["ativacao"] == 0.0  # a fala livre não entra nas frases NDD
    f = s["NDD_LIVRE"]
    assert f["n"] == 1 and f["falas"] == 4 and f["minutos"] == pytest.approx(1.0)
    assert f["fp_por_fala"] == 0.25 and f["fp_por_fala_politica_baixa_ndd"] == 0.0
    assert f["ativacoes_por_minuto"] == pytest.approx(1.0)


def test_tts_cache_name_depends_on_text_and_voice() -> None:
    from ddsd.evaluation.asr_impact import tts_cache_name

    a = tts_cache_name("abre o navegador", "pt-BR-FranciscaNeural")
    assert a == tts_cache_name("abre o navegador", "pt-BR-FranciscaNeural")  # estável entre execuções
    assert a != tts_cache_name("abre o navegador", "pt-BR-AntonioNeural")
    assert a != tts_cache_name("fecha o navegador", "pt-BR-FranciscaNeural")
    assert a.startswith("tts_") and a.endswith(".mp3") and len(a) == len("tts_") + 16 + len(".mp3")


def test_internal_report_helpers() -> None:
    from ddsd.evaluation.commands import _bin_order, _mean_sd

    assert sorted(["11-13", "2", "7-8", "25-50", "geral"], key=_bin_order) == ["2", "7-8", "11-13", "25-50", "geral"]
    assert _mean_sd(0.97701, float("nan")) == "0.9770"
    assert _mean_sd(0.97701, 0.00123) == "0.9770 ± 0.0012"


def test_seed_summary_has_no_spread_with_one_seed(tmp_path) -> None:
    import json
    import math

    from ddsd.evaluation.runs import Run, best_seed, seed_summary

    def run(seed: int, acc: float) -> Run:
        path = tmp_path / f"m-s{seed}"
        path.mkdir()
        result = {"val_metrics": {"f1_macro": 1 - acc}, "test_metrics": {"accuracy": acc}}
        (path / "run_manifest.json").write_text(json.dumps({"result": result}), encoding="utf-8")
        return Run("principal", "m", seed, path)

    one = seed_summary([run(42, 0.9)], "principal", "m", ("accuracy",))
    assert one["n_sementes"] == 1 and math.isnan(one["accuracy"]["desvio"])
    runs = [run(43, 0.92), run(44, 0.94)]
    three = seed_summary([*runs, Run("principal", "m", 42, tmp_path / "m-s42")], "principal", "m", ("accuracy",))
    assert three["accuracy"]["media"] == pytest.approx(0.92) and three["accuracy"]["desvio"] == pytest.approx(0.02)
    # A semente sai da validação, nunca do teste: aqui a maior F1 de validação é a de pior teste.
    assert best_seed(runs, "principal", "m").seed == 43


def _tiny_corpus(directory) -> None:
    import json
    import random

    rng = random.Random(0)
    dd = ["abre o navegador", "fecha a aba", "aumenta o volume", "abre o editor de texto", "pausa a musica",
          "salva o arquivo", "abre a planilha", "procura receita de bolo", "minimiza a janela", "tira um print"]
    ndd = ["vamos almocar", "passa o sal", "que dia lindo", "me liga depois", "ele chegou tarde",
           "fecha a porta do carro", "traz um copo de agua", "a reuniao atrasou", "gostei do filme", "segura isso"]
    directory.mkdir(parents=True)
    for split, n in (("train", 60), ("val", 20), ("test", 20)):
        rows = []
        for i in range(n):
            label, pool = ("DD", dd) if i % 2 == 0 else ("NDD", ndd)
            text = f"{rng.choice(pool)} {rng.choice(['agora', 'ja', 'por favor', 'aqui', 'logo'])}"
            rows.append({"text": text, "label": label, "source": "s", "origin": "o", "bin": "3",
                         "words": len(text.split())})
        (directory / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_baseline_classifier_matches_training_predictions(tmp_path) -> None:
    import json

    from ddsd.classifiers.model import ArtifactError
    from ddsd.evaluation.baselines import MODEL_FILE, BaselineClassifier, train_baseline

    _tiny_corpus(tmp_path / "splits")
    train_baseline("logreg", tmp_path / "splits", tmp_path / "models")
    run_dir = tmp_path / "models" / "baseline_logreg"
    clf = BaselineClassifier(run_dir)
    saved = [json.loads(line) for line in (run_dir / "predictions_test.jsonl").read_text().splitlines()]
    preds = clf.predict_batch([r["text"] for r in saved])
    assert [p.label for p in preds] == [r["pred"] for r in saved]
    assert max(abs(p.p_dd - r["score_dd"]) for p, r in zip(preds, saved, strict=True)) < 1e-6  # 6 casas no arquivo
    single = clf.predict(saved[0]["text"])
    assert single.label == preds[0].label and single.latency_ms > 0

    with (run_dir / MODEL_FILE).open("ab") as handle:  # artefato adulterado
        handle.write(b"x")
    with pytest.raises(ArtifactError):
        BaselineClassifier(run_dir)


def test_paired_fp_bootstrap_detects_lower_fp_and_misalignment() -> None:
    from ddsd.evaluation.external_text import paired_fp_bootstrap

    a = ["DD"] * 10 + ["NDD"] * 90
    b = ["DD"] * 30 + ["NDD"] * 70
    result = paired_fp_bootstrap(a, b, n_bootstrap=500)
    assert result["fp_a"] == pytest.approx(0.1) and result["diferenca"] == pytest.approx(-0.2)
    assert result["ic95_sup"] < 0 and result["p_a_menos_fp"] > 0.99
    with pytest.raises(ValueError):
        paired_fp_bootstrap(a, b[:-1])


def test_classifier_benchmark_measures_one_classifier_at_a_time(monkeypatch) -> None:
    """Um classificador por vez, como no pipeline (no MPS, inferência concorrente aborta o processo no Metal)."""
    import threading
    import time
    from types import SimpleNamespace

    import torch

    import ddsd.classifiers.model as model_module
    from ddsd.evaluation.benchmark import bench_classifiers

    lock = threading.Lock()
    active = {"cpu": 0, "mps": 0}
    peak = {"cpu": 0, "mps": 0}

    class FakeClassifier:
        def __init__(self, name, path, device="cpu"):
            self.device = device

        def warm_up(self) -> None:
            pass

        def predict(self, text):
            with lock:
                active[self.device] += 1
                peak[self.device] = max(peak[self.device], active[self.device])
            time.sleep(0.002)  # janela para as threads se sobreporem
            with lock:
                active[self.device] -= 1
            return SimpleNamespace(latency_ms=2.0)

    monkeypatch.setattr(model_module, "DddClassifier", FakeClassifier)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    threads_before = torch.get_num_threads()
    try:
        out = bench_classifiers({"a": None, "b": None, "c": None}, repeats=8)
    finally:
        torch.set_num_threads(threads_before)
    assert set(out) == {"cpu-1t", "cpu-2t", "cpu-4t", "mps-2t"}
    assert all(set(out[k]) == {"isolado"} and set(out[k]["isolado"]) == {"a", "b", "c"} for k in out)
    assert peak["mps"] == 1 and peak["cpu"] == 1


def test_safetensors_parameter_count_reads_only_the_header(tmp_path) -> None:
    import json
    import struct

    from ddsd.evaluation.cost import safetensors_params

    header = {"__metadata__": {"format": "pt"}, "a": {"dtype": "F32", "shape": [3, 4], "data_offsets": [0, 48]},
              "b": {"dtype": "F32", "shape": [5], "data_offsets": [48, 68]}}
    raw = json.dumps(header).encode()
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 68)
    assert safetensors_params(path) == 3 * 4 + 5


def test_paired_asr_drop_bootstrap_detects_the_more_fragile_model() -> None:
    from ddsd.evaluation.asr_impact import paired_drop_bootstrap

    records = []
    for i in range(200):
        label = "DD" if i % 2 else "NDD"
        wrong = "NDD" if label == "DD" else "DD"
        records.append({"wer": 0.2, "label": label,
                        "a:reference_norm": label, "a:hypothesis": label,                    # A não perde nada
                        "b:reference_norm": label, "b:hypothesis": wrong if i < 40 else label})  # B perde 20%
    records.append({"wer": None, "label": "NDD", "a:reference_norm": "NDD", "a:hypothesis": "DD",  # fala livre: fora
                    "b:reference_norm": "NDD", "b:hypothesis": "NDD"})
    r = paired_drop_bootstrap(records, "a", "b", n_bootstrap=300)
    assert r["n"] == 200 and r["queda_a"] == 0.0 and r["queda_b"] == pytest.approx(0.2)
    assert r["ic95_sup"] < 0 and r["p_a_perde_menos"] == 1.0


def test_pipeline_items_for_recordings_and_synthetic_voice(tmp_path) -> None:
    import json

    from ddsd.evaluation import pipeline_eval

    manifest = tmp_path / "manifest.jsonl"
    rows = [{"scenario": "quieto_interno", "item": "P00", "file": "a.wav", "reference": "abre o navegador",
             "label": "DD"},
            {"scenario": "quieto_interno", "item": "fala_livre", "file": "b.wav", "reference": "", "label": "NDD"},
            {"scenario": "quieto_interno", "item": "ruido", "file": "c.wav", "reference": "", "label": "RUIDO"},
            {"scenario": "conversa_interno", "item": "C01", "file": "d.wav", "reference": "bom dia", "label": "NDD"}]
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    items = pipeline_eval.recording_items(manifest, tmp_path)
    assert [(i.item_id, i.label, i.group) for i in items] == [
        ("quieto_interno/P00", "DD", "quieto_interno"), ("quieto_interno/fala_livre", "NDD_LIVRE", "quieto_interno"),
        ("quieto_interno/ruido", "RUIDO", "quieto_interno"), ("conversa_interno/C01", "NDD", "conversa_interno")]
    assert items[0].path == tmp_path / "a.wav" and items[3].reference == "bom dia"
    (tmp_path / "itens.jsonl").write_text(json.dumps({"id": "tts_0000", "arquivo": "x.mp3", "texto": "Cadê meu café?",
                                                      "rotulo": "NDD", "origem": "tts:voz"}) + "\n")
    tts = pipeline_eval.tts_items(tmp_path)
    assert [(i.item_id, i.path.name, i.reference, i.label, i.group) for i in tts] == [
        ("tts_0000", "x.mp3", "Cadê meu café?", "NDD", "tts:voz")]


def test_text_predictions_feed_the_paired_bootstrap_on_the_transcriptions() -> None:
    from ddsd.evaluation.asr_impact import text_predictions
    from ddsd.evaluation.bootstrap import pairwise

    records = []
    for i in range(120):
        label = "DD" if i % 2 else "NDD"
        wrong = "NDD" if label == "DD" else "DD"
        records.append({"wer": 0.1, "label": label, "hypothesis": f"frase {i}", "reference_norm": f"frase {i}",
                        "a:hypothesis": label, "a:hypothesis:p": 0.9 if label == "DD" else 0.1,
                        "b:hypothesis": wrong if i < 30 else label, "b:hypothesis:p": 0.5})
    records.append({"wer": 0.0, "label": "DD", "hypothesis": "", "reference_norm": "x",  # sem texto: NDD, p = 0
                    "a:hypothesis": "NDD", "b:hypothesis": "NDD"})
    records.append({"wer": None, "label": "NDD", "hypothesis": "fala livre", "reference_norm": "",  # fora
                    "a:hypothesis": "DD", "a:hypothesis:p": 0.9, "b:hypothesis": "NDD", "b:hypothesis:p": 0.1})
    preds = text_predictions(records, ["a", "b"])
    assert len(preds["a"]["y_true"]) == 121 and preds["a"]["y_score"][-1] == 0.0
    result = pairwise(preds, n_bootstrap=200)["a vs b"]
    assert result["accuracy"]["ic95_inf"] > 0 and result["eer"]["ic95_sup"] < 0


def test_resampler_draws_whole_sentences() -> None:
    from collections import Counter

    from ddsd.evaluation.resampling import make_resampler

    groups = ["a", "b", "a", "c", "b", "a"]
    draw = make_resampler(len(groups), groups)
    rng = np.random.default_rng(0)
    for _ in range(50):
        counts = Counter(groups[i] for i in draw(rng))
        assert all(c % groups.count(g) == 0 for g, c in counts.items())   # a frase entra com todas as leituras
        assert sum(c // groups.count(g) for g, c in counts.items()) == 3  # 3 frases sorteadas com reposição
    assert len(make_resampler(6)(np.random.default_rng(0))) == 6
    with pytest.raises(ValueError):
        make_resampler(5, groups)


def test_sentence_bootstrap_widens_the_interval_when_errors_repeat() -> None:
    from ddsd.evaluation.external_text import paired_fp_bootstrap
    from ddsd.evaluation.resampling import group_field, unit_name

    groups = [f"s{i}" for _ in range(4) for i in range(10)]       # 10 frases lidas em 4 cenários
    a = ["DD" if g in ("s0", "s1") else "NDD" for g in groups]   # A erra sempre nas mesmas 2 frases
    b = ["NDD"] * len(groups)
    by_item = paired_fp_bootstrap(a, b, n_bootstrap=1000)
    by_sentence = paired_fp_bootstrap(a, b, n_bootstrap=1000, groups=groups)
    assert by_item["diferenca"] == by_sentence["diferenca"] == pytest.approx(0.2)
    assert by_sentence["ic95_inf"] <= 0.0 < by_item["ic95_inf"]
    assert group_field("gravacoes") == "reference"
    assert group_field("tts") is None and group_field("publico") is None
    assert unit_name(group_field("gravacoes")) == "frase" and unit_name(None) == "item"


def test_drop_and_metric_bootstraps_accept_sentence_groups() -> None:
    from ddsd.evaluation.asr_impact import paired_drop_bootstrap
    from ddsd.evaluation.bootstrap import pairwise

    records = [{"wer": 0.1, "label": "DD", "reference": f"frase {i % 10}",
                "a:reference_norm": "DD", "a:hypothesis": "DD",
                "b:reference_norm": "DD", "b:hypothesis": "NDD" if i % 10 < 2 else "DD"} for i in range(40)]
    by_item = paired_drop_bootstrap(records, "a", "b", n_bootstrap=500)
    by_sentence = paired_drop_bootstrap(records, "a", "b", n_bootstrap=500, group_field="reference")
    assert by_item["diferenca"] == by_sentence["diferenca"] == pytest.approx(-0.2)
    assert by_item["ic95_sup"] < 0.0 <= by_sentence["ic95_sup"]

    y = [1] * 20 + [0] * 20
    groups = [f"s{i % 10}" for i in range(20)] + [f"n{i % 10}" for i in range(20)]
    good = preds(y, y, [0.9] * 20 + [0.1] * 20)
    bad = preds(y, [0 if i % 10 < 2 else 1 for i in range(20)] + [0] * 20,
                [0.4 if i % 10 < 2 else 0.9 for i in range(20)] + [0.1] * 20)
    item = pairwise({"a": good, "b": bad}, n_bootstrap=300)["a vs b"]["accuracy"]
    sentence = pairwise({"a": good, "b": bad}, n_bootstrap=300, groups=groups)["a vs b"]["accuracy"]
    assert item["diferenca_media"] == pytest.approx(0.1, abs=0.02)
    assert sentence["ic95_inf"] < item["ic95_inf"]
