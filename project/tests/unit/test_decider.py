"""Decisor (classificador ativo, regra tabular) e agregador (isolamento, prazos, atrasos)."""

from __future__ import annotations

import random

import pytest

from ddsd.asr.types import Transcription, WhisperSegment
from ddsd.classifiers.model import Prediction
from ddsd.decision.decider import EvidenceAggregator, TaskEvidence, TaskState, decide
from ddsd.fuzzy.reliability import Reliability
from ddsd.realtime.domain import AudioSpan, CloseReason, TaskId

SPAN = AudioSpan(0, 16_000)
ACTIVE = "bertimbau"


def pred(model: str, label: str, p: float = 0.9, threshold: float = 0.5) -> Prediction:
    return Prediction(model, p if label == "DD" else 1 - p, threshold, label, 10.0, 5, False)


def transcription(text: str) -> Transcription:
    seg = WhisperSegment(0.0, 1.0, text, (1, 2), -0.2, 1.2, 0.05, 0.0, 0)
    return Transcription(text, "pt", (seg,), "mlx", "m", 1.0, 0.3)


def evidence(task_id: TaskId, labels: dict[str, str], text: str = "abre o navegador",
             reliability: float = 0.8, rel_label: str = "alta") -> TaskEvidence:
    e = TaskEvidence(task_id, SPAN, TaskState.DONE, transcription(text), None, text,
                     Reliability(reliability, rel_label, None))
    for model, label in labels.items():
        e.predictions[model] = pred(model, label)
    return e


FINAL = TaskId(1)


def run_decide(final: TaskEvidence, classifier: str = ACTIVE):
    return decide(final, classifier=classifier, utterance=1, close_reason="fim_da_fala", speech_ended_at=0.0,
                  now=1.0, timed_out=False)


@pytest.mark.parametrize("label", ["DD", "NDD"])
def test_the_active_classifier_label_is_the_decision(label: str) -> None:
    d = run_decide(evidence(FINAL, {ACTIVE: label}))
    assert (d.label, d.reason, d.classifier) == (label, "classificador", ACTIVE)
    assert set(d.votes) == {ACTIVE} and d.votes[ACTIVE][0] == label


def test_only_the_active_classifier_counts() -> None:
    # As outras execuções são alternativas: um resultado delas não decide nem entra nos votos.
    final = evidence(FINAL, {"albertina": "DD", "debertinha": "DD"})
    assert (run_decide(final).label, run_decide(final).reason) == ("NDD", "falha_classificador")
    d = run_decide(final, classifier="albertina")
    assert (d.label, d.classifier, set(d.votes)) == ("DD", "albertina", {"albertina"})


def test_empty_text_and_asr_failure_are_ndd() -> None:
    assert run_decide(evidence(FINAL, {}, text="")).reason == "sem_texto"
    failed = TaskEvidence(FINAL, SPAN, TaskState.FAILED, None, "erro", None, None)
    assert (run_decide(failed).label, run_decide(failed).reason) == ("NDD", "falha_asr")


def test_low_reliability_flags_but_never_flips_label() -> None:
    d = run_decide(evidence(FINAL, {ACTIVE: "DD"}, reliability=0.2, rel_label="baixa"))
    assert d.label == "DD" and d.reliability_alert


def test_decision_keeps_the_utterance_evidence_and_latency() -> None:
    final = evidence(FINAL, {ACTIVE: "DD"}, reliability=0.9)
    d = decide(final, classifier=ACTIVE, utterance=1, close_reason="fim_da_fala", speech_ended_at=2.0, now=3.25,
               timed_out=False)
    assert d.evidence == (final,) and d.latency_s == pytest.approx(1.25)
    assert d.to_dict()["tasks"][0]["task"] == "U0001-F"


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def feed_utterance(u: int, label: str) -> tuple[list, tuple, list]:
    """Eventos completos de uma fala, para embaralhar com os de outras falas."""
    final = TaskId(u)
    tail = [("state", final, TaskState.ASR), ("asr", final, f"final {u}"), ("clf", final, ACTIVE, label),
            ("fuzzy", final)]
    return [("open", u)], ("close", final), tail


def apply(agg: EvidenceAggregator, event: tuple) -> None:
    kind = event[0]
    if kind == "open":
        agg.utterance_opened(event[1], 0.0)
    elif kind == "close":
        agg.utterance_closed(event[1], SPAN, CloseReason.END_OF_SPEECH, 0.0)
    elif kind == "state":
        agg.task_state(event[1], event[2])
    elif kind == "asr":
        agg.transcription(event[1], transcription(event[2]), None, event[2])
    elif kind == "clf":
        agg.component_result(event[1], event[2], pred(event[2], event[3]))
    elif kind == "fuzzy":
        agg.component_result(event[1], "fuzzy", Reliability(0.9, "alta", None))


def test_interleaved_utterances_never_mix() -> None:
    decisions = []
    agg = EvidenceAggregator(classifier=ACTIVE, on_decision=decisions.append)
    rng = random.Random(7)
    expected = {}
    streams = []
    for u in range(1, 9):
        label = rng.choice(["DD", "NDD"])
        expected[u] = label
        head, close, tail = feed_utterance(u, label)
        for e in head:
            apply(agg, e)
        apply(agg, close)
        streams.append(tail)
    # Intercala aleatoriamente os resultados de todas as falas (mantendo a ordem dentro de cada tarefa).
    while any(streams):
        stream = rng.choice([s for s in streams if s])
        apply(agg, stream.pop(0))
    assert {d.utterance: d.label for d in decisions} == expected
    for d in decisions:
        assert d.text_raw == f"final {d.utterance}"
        assert all(e.task_id.utterance == d.utterance for e in d.evidence)
    assert agg.rejected_events == 0 and not agg.pending()


def test_inconsistent_keys_are_rejected() -> None:
    agg = EvidenceAggregator(classifier=ACTIVE)
    agg.component_result(TaskId(99), ACTIVE, pred(ACTIVE, "DD"))
    assert agg.rejected_events == 1
    agg.utterance_opened(1, 0.0)
    agg.utterance_closed(FINAL, SPAN, CloseReason.END_OF_SPEECH, 0.0)
    agg.utterance_closed(FINAL, SPAN, CloseReason.END_OF_SPEECH, 0.0)  # fechada duas vezes
    agg.transcription(FINAL, transcription("x"), None, "x")
    agg.component_result(FINAL, ACTIVE, pred("albertina", "DD"))  # modelo trocado
    agg.component_result(FINAL, ACTIVE, pred(ACTIVE, "DD"))
    agg.component_result(FINAL, ACTIVE, pred(ACTIVE, "DD"))  # repetido
    assert agg.rejected_events == 4


def test_timeout_counts_from_final_transcription_and_late_results_are_warnings() -> None:
    clock = Clock()
    decisions = []
    agg = EvidenceAggregator(classifier=ACTIVE, timeout_s=5.0, asr_timeout_s=60.0, on_decision=decisions.append,
                             clock=clock)
    agg.utterance_opened(1, 0.0)
    agg.utterance_closed(FINAL, SPAN, CloseReason.END_OF_SPEECH, 0.0)
    clock.t = 30.0  # ASR lento: ainda dentro do prazo do ASR
    agg.check_timeouts()
    assert not decisions
    agg.transcription(FINAL, transcription("abre"), None, "abre")
    agg.component_result(FINAL, "fuzzy", Reliability(0.9, "alta", None))
    clock.t = 34.0
    agg.check_timeouts()
    assert not decisions  # 4 s depois da transcrição: ainda espera o classificador
    clock.t = 35.5
    agg.check_timeouts()
    assert decisions and decisions[0].timed_out and decisions[0].reason == "falha_classificador"
    agg.component_result(FINAL, ACTIVE, pred(ACTIVE, "DD"))  # chegou atrasado
    assert agg.late_events == 1 and agg.rejected_events == 0
