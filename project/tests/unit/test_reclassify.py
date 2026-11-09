"""Comparação pareada no pipeline: regra do decisor reaproveitada, paridade e alinhamento."""

from __future__ import annotations

import pytest

from ddsd.asr.types import Transcription, WhisperSegment
from ddsd.classifiers.model import Prediction
from ddsd.decision.decider import TaskEvidence, TaskState, decide
from ddsd.evaluation import reclassify
from ddsd.fuzzy.reliability import Reliability
from ddsd.realtime.domain import AudioSpan, TaskId


class FakeClassifier:
    """DD quando o texto tem a palavra "abre" (ou o contrário, para quebrar a paridade)."""

    def __init__(self, name: str, flip: bool = False) -> None:
        self.name, self.flip, self.calls = name, flip, 0

    def predict(self, text: str) -> Prediction:
        self.calls += 1
        dd = ("abre" in text.split()) != self.flip
        return Prediction(self.name, 0.9 if dd else 0.1, 0.5, "DD" if dd else "NDD", 1.5, len(text.split()), False)


def decision(u: int, text: str, *, rel: str = "alta", asr_error: str | None = None) -> dict:
    """Decisão real do decisor (mesmo formato dos registros do pipeline), feita com o BERTimbau falso."""
    seg = WhisperSegment(0.0, 1.0, text, (1,), -0.2, 1.2, 0.05, 0.0, 0)
    tr = None if asr_error else Transcription(text, "pt", (seg,), "mlx", "m", 1.0, 0.3)
    e = TaskEvidence(TaskId(u), AudioSpan(0, 16_000), TaskState.DONE if not asr_error else TaskState.FAILED, tr,
                     asr_error, None if asr_error else text, Reliability(0.3 if rel == "baixa" else 0.8, rel, None))
    if not asr_error and text:
        e.predictions["bertimbau"] = FakeClassifier("bertimbau").predict(text)
    return decide(e, classifier="bertimbau", utterance=u, close_reason="fim_da_fala", speech_ended_at=0.0,
                  now=1.2, timed_out=False).to_dict()


def record(item: str, label: str, decisions: list[dict], group: str = "g") -> dict:
    return {"id": item, "group": group, "label": label, "reference": "", "classifier": "bertimbau",
            "audio_s": 2.0, "utterances": len(decisions), "activated": any(d["label"] == "DD" for d in decisions),
            "activated_low_policy": any(d["label"] == "DD" and not d["reliability"]["alert"] for d in decisions),
            "decisions": decisions, "latencies_s": [1.2] * len(decisions)}


RECORDS = [
    record("a", "DD", [decision(1, "abre o navegador")]),
    record("b", "NDD", [decision(2, "passa o sal")]),
    record("c", "NDD", [decision(3, "abre a janela pra mim", rel="baixa")]),  # FP com alerta do fuzzy
    record("d", "NDD", [decision(4, "")]),                                     # sem texto
    record("e", "DD", [decision(5, "x", asr_error="falha")]),                  # ASR falhou
    record("f", "NDD", []),                                                    # nenhuma fala detectada
]


def test_reclassifying_with_the_run_classifier_reproduces_every_decision() -> None:
    clf = FakeClassifier("bertimbau")
    out, inference = reclassify.reclassify(RECORDS, "bertimbau", clf)
    parity = reclassify.check_parity(RECORDS, out, "bertimbau")
    assert parity == {"falas": 5, "identicas": 5, "max_delta_p": 0.0}
    assert clf.calls == 3 and len(inference) == 3  # sem texto e falha do ASR não chamam o classificador
    reasons = [d["reason"] for r in out for d in r["decisions"]]
    assert reasons == ["classificador", "classificador", "classificador", "sem_texto", "falha_asr"]
    assert [r["activated"] for r in out] == [True, False, True, False, False, False]
    assert [r["activated_low_policy"] for r in out] == [True, False, False, False, False, False]
    assert all(r["latencies_s"] == [] for r in out)  # a latência ponta a ponta não é atribuída a outro modelo


def test_parity_failure_stops_the_comparison() -> None:
    out, _ = reclassify.reclassify(RECORDS, "bertimbau", FakeClassifier("bertimbau", flip=True))
    with pytest.raises(reclassify.ParityError, match="3 de 5"):
        reclassify.check_parity(RECORDS, out, "bertimbau")


def test_compare_is_paired_and_rejects_misaligned_items() -> None:
    by = {name: reclassify.reclassify(RECORDS, name, FakeClassifier(name, flip=name != "bertimbau"))[0]
          for name in ("bertimbau", "baseline_svm")}
    result = reclassify.compare(by, n_bootstrap=200)
    assert result["por_grupo"]["bertimbau"]["g"]["NDD"] == {"n": 4, "ativacao": 0.25,
                                                            "ativacao_politica_baixa_ndd": 0.0}
    pair = result["pareado"]["baseline_svm vs bertimbau"]["NDD"]
    assert pair["fp_b"] == 0.25 and pair["fp_a"] == 0.25  # "passa o sal" vira DD; "abre a janela", NDD
    by["baseline_svm"] = list(reversed(by["baseline_svm"]))
    with pytest.raises(ValueError, match="desalinhados"):
        reclassify.compare(by, n_bootstrap=10)


def test_default_pairs_compare_each_model_with_bertimbau_and_with_its_own_control() -> None:
    names = ["bertimbau", "baseline_svm", "albertina", "sem_correcao_bertimbau", "sem_correcao_albertina"]
    assert reclassify.default_pairs(names) == [
        ("baseline_svm", "bertimbau"), ("albertina", "bertimbau"),
        ("sem_correcao_bertimbau", "bertimbau"), ("sem_correcao_albertina", "albertina")]


def test_same_transcriptions_counts_items_with_identical_spans_and_text() -> None:
    other = [dict(r) for r in RECORDS]
    changed = dict(other[1])
    changed["decisions"] = [dict(changed["decisions"][0])]
    changed["decisions"][0]["tasks"] = [dict(t, text_raw="passa a sal") for t in changed["decisions"][0]["tasks"]]
    other[1] = changed
    out = reclassify.same_transcriptions(RECORDS, other)
    assert out == {"itens": 6, "segmentacao_identica": 6, "transcricao_identica": 5, "itens_diferentes": ["b"]}
    assert reclassify.same_transcriptions(RECORDS, RECORDS)["transcricao_identica"] == 6
    with pytest.raises(ValueError, match="mesmos itens"):
        reclassify.same_transcriptions(RECORDS, RECORDS[:-1])


def test_inference_time_is_read_from_the_live_run_predictions() -> None:
    assert reclassify.inference_ms(RECORDS, "bertimbau") == [1.5, 1.5, 1.5]
    assert reclassify.inference_ms(RECORDS, "albertina") == []


def test_error_origin_separates_segmentation_classifier_and_transcription() -> None:
    from ddsd.evaluation.error_origin import error_origins

    def item(i: str, label: str, activated: bool, utterances: int = 1) -> dict:
        return {"id": i, "label": label, "activated": activated, "utterances": utterances}

    records = [item("ok", "DD", True), item("seg", "DD", False, utterances=0), item("split", "NDD", True, 2),
               item("txt", "NDD", True), item("asr", "DD", False), item("free", "NDD_LIVRE", True)]
    asr = [{"id": "txt", "m:reference_norm": "DD"}, {"id": "asr", "m:reference_norm": "DD"}]
    out = error_origins(records, asr, "m")
    assert out["por_rotulo"] == {"DD": {"segmentacao": 1, "classificador": 0, "transcricao": 1},
                                 "NDD": {"segmentacao": 1, "classificador": 1, "transcricao": 0}}
    assert out["itens"] == {"segmentacao": ["seg", "split"], "classificador": ["txt"], "transcricao": ["asr"]}


def test_compare_draws_whole_sentences_when_asked() -> None:
    def records(active) -> list[dict]:  # 10 frases NDD lidas em 4 cenários
        return [{"id": f"c{k}/{s}", "group": f"c{k}", "label": "NDD", "reference": f"frase {s}",
                 "activated": active(s), "activated_low_policy": active(s)} for k in range(4) for s in range(10)]

    by = {"bertimbau": records(lambda s: False), "baseline_svm": records(lambda s: s < 2)}
    key = "baseline_svm vs bertimbau"
    item = reclassify.compare(by, n_bootstrap=500)["pareado"][key]["NDD"]
    sentence = reclassify.compare(by, n_bootstrap=500, group_field="reference")["pareado"][key]["NDD"]
    assert item["diferenca"] == sentence["diferenca"] == pytest.approx(0.2)
    assert sentence["ic95_inf"] <= 0.0 < item["ic95_inf"]
