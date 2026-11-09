"""Agregação de evidências por fala e decisão DD/NDD.

Regra de decisão:
1. O pipeline usa um classificador por vez (BERTimbau, Albertina, DeBERTinha, TF-IDF + Regressão
   Logística ou TF-IDF + SVM, escolhido em `[classifiers] active`; padrão BERTimbau). Os cinco são
   alternativas, não um comitê. O rótulo do
   classificador ativo sobre a transcrição final, com o limiar congelado da execução, é a decisão
   da fala. O classificador é fixado por fala na abertura: uma troca no painel vale a partir da
   fala seguinte, e uma fala nunca mistura modelos.
2. A confiabilidade fuzzy (r) acompanha a decisão. Confiabilidade baixa gera
   `alerta_confiabilidade`, mas não muda o rótulo. A política "baixa → NDD" só é avaliada offline.
3. Falhas: sem texto → NDD (`sem_texto`); ASR final com erro → NDD (`falha_asr`); classificador sem
   resposta (erro ou prazo) → NDD (`falha_classificador`).

O agregador indexa tudo por (fala, componente). Uma evidência com chave
inconsistente (fala desconhecida, componente repetido, modelo trocado) é
rejeitada e registrada; nunca se mistura com as de outra fala.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from ddsd.asr.types import Transcription
from ddsd.classifiers.model import Prediction
from ddsd.fuzzy.reliability import Reliability
from ddsd.observability.logger import get_logger
from ddsd.realtime.domain import AudioSpan, CloseReason, TaskId

log = get_logger("decisor")

#: Classificador padrão do pipeline.
DEFAULT_CLASSIFIER = "bertimbau"


class TaskState(StrEnum):
    QUEUED = "na_fila"
    ASR = "transcrevendo"
    NLP = "analisando"
    DONE = "concluida"
    FAILED = "falhou"


@dataclass
class TaskEvidence:
    """Tudo o que foi produzido para a tarefa de uma fala (o processamento da fala completa)."""

    task_id: TaskId
    span: AudioSpan
    state: TaskState = TaskState.QUEUED
    transcription: Transcription | None = None
    asr_error: str | None = None
    normalized: str | None = None
    reliability: Reliability | None = None
    predictions: dict[str, Prediction] = field(default_factory=dict)
    component_errors: dict[str, str] = field(default_factory=dict)
    skipped: set[str] = field(default_factory=set)

    def components_pending(self, expected: tuple[str, ...]) -> list[str]:
        done = set(self.predictions) | set(self.component_errors) | self.skipped
        pending = [name for name in expected if name not in done]
        if self.reliability is None and "fuzzy" not in self.component_errors and "fuzzy" not in self.skipped:
            pending.append("fuzzy")
        return pending

    def label_by(self, classifier: str) -> tuple[str, str]:
        """(rótulo, motivo) desta tarefa pela regra do decisor, com o classificador ativo."""
        if self.asr_error is not None:
            return "NDD", "falha_asr"
        if not self.normalized:
            return "NDD", "sem_texto"
        prediction = self.predictions.get(classifier)
        if prediction is None:
            return "NDD", "falha_classificador"
        return prediction.label, "classificador"


@dataclass(frozen=True)
class FinalDecision:
    """Decisão de uma fala, com as evidências individuais preservadas para rastreabilidade."""

    utterance: int
    label: str
    reason: str
    classifier: str
    votes: dict[str, tuple[str, float, float]]
    reliability_score: float | None
    reliability_label: str | None
    reliability_alert: bool
    text_raw: str
    normalized: str
    span: AudioSpan
    close_reason: str
    latency_s: float | None
    decided_at: float
    timed_out: bool
    evidence: tuple[TaskEvidence, ...]

    @property
    def utterance_label(self) -> str:
        return f"U{self.utterance:04d}"

    def to_dict(self) -> dict[str, object]:
        """Resumo serializável (sem o áudio)."""
        def task(e: TaskEvidence) -> dict[str, object]:
            tr = e.transcription
            return {
                "task": str(e.task_id), "span_s": e.span.seconds(), "state": e.state.value,
                "text_raw": tr.text_raw if tr else None, "normalized": e.normalized,
                "asr_latency_s": round(tr.latency_s, 3) if tr else None, "asr_error": e.asr_error,
                "segments": [dataclasses.asdict(s) for s in (tr.segments if tr else ())],
                "reliability": None if e.reliability is None else {
                    "score": round(e.reliability.score, 4), "label": e.reliability.label,
                    "inputs": e.reliability.inputs.as_dict() if e.reliability.inputs else None},
                "predictions": {m: {"label": p.label, "p_dd": round(p.p_dd, 6), "threshold": p.threshold,
                                    "latency_ms": round(p.latency_ms, 2)} for m, p in e.predictions.items()},
                "errors": e.component_errors, "skipped": sorted(e.skipped),
            }

        return {
            "utterance": self.utterance_label, "label": self.label, "reason": self.reason,
            "classifier": self.classifier,
            "votes": {m: {"label": l, "p_dd": round(p, 6), "threshold": t} for m, (l, p, t) in self.votes.items()},
            "reliability": {"score": self.reliability_score, "label": self.reliability_label,
                            "alert": self.reliability_alert},
            "text_raw": self.text_raw,
            "normalized": self.normalized, "span_s": self.span.seconds(), "close_reason": self.close_reason,
            "latency_s": None if self.latency_s is None else round(self.latency_s, 3), "timed_out": self.timed_out,
            "tasks": [task(e) for e in self.evidence],
        }


def decide(final: TaskEvidence, *, classifier: str, utterance: int, close_reason: str,
           speech_ended_at: float | None, now: float, timed_out: bool) -> FinalDecision:
    """Aplica a regra de decisão às evidências da fala completa (classificador ativo + fuzzy)."""
    text_raw = final.transcription.text_raw if final.transcription else ""
    normalized = final.normalized or ""
    votes = {m: (p.label, p.p_dd, p.threshold) for m, p in final.predictions.items() if m == classifier}
    rel = final.reliability
    label, reason = final.label_by(classifier)
    return FinalDecision(
        utterance=utterance, label=label, reason=reason, classifier=classifier, votes=votes,
        reliability_score=None if rel is None else round(rel.score, 4),
        reliability_label=None if rel is None else rel.label,
        reliability_alert=rel is not None and rel.label == "baixa",
        text_raw=text_raw, normalized=normalized, span=final.span, close_reason=close_reason,
        latency_s=None if speech_ended_at is None else now - speech_ended_at, decided_at=now,
        timed_out=timed_out, evidence=(final,),
    )


@dataclass
class UtteranceRecord:
    utterance: int
    opened_at: float
    classifier: str = DEFAULT_CLASSIFIER
    tasks: dict[TaskId, TaskEvidence] = field(default_factory=dict)
    closed: bool = False
    close_reason: str = ""
    speech_ended_at: float | None = None
    final_enqueued_at: float | None = None
    final_transcribed_at: float | None = None
    decided: bool = False


class EvidenceAggregator:
    """Reúne as evidências de cada fala e dispara a decisão quando estão completas."""

    def __init__(self, *, classifier: str = DEFAULT_CLASSIFIER, timeout_s: float = 5.0,
                 asr_timeout_s: float = 60.0, on_decision: Callable[[FinalDecision], None] | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        #: Classificador das falas abertas daqui em diante (cada fala guarda o seu).
        self.classifier = classifier
        #: Prazo das análises (fuzzy e classificadores) contado do FIM da transcrição final.
        self._timeout = timeout_s
        #: Prazo de segurança para o próprio ASR (travamento), contado do enfileiramento da final.
        self._asr_timeout = asr_timeout_s
        #: Falas já decididas (limitado), para distinguir resultado atrasado de chave inconsistente.
        self._decided: deque[int] = deque(maxlen=1000)
        self._on_decision = on_decision or (lambda d: None)
        self._clock = clock
        self._records: dict[int, UtteranceRecord] = {}
        self._lock = threading.RLock()
        self.rejected_events = 0
        self.late_events = 0

    # ---------------------------------------------------------------- eventos
    def set_classifier(self, classifier: str) -> None:
        """Troca o classificador das próximas falas; as já abertas terminam com o delas."""
        with self._lock:
            self.classifier = classifier

    def classifier_of(self, utterance: int) -> str:
        with self._lock:
            record = self._records.get(utterance)
            return record.classifier if record is not None else self.classifier

    def utterance_opened(self, utterance: int, opened_at: float) -> None:
        with self._lock:
            self._records.setdefault(utterance, UtteranceRecord(utterance, opened_at, self.classifier))

    def task_state(self, task_id: TaskId, state: TaskState) -> None:
        with self._lock:
            evidence = self._evidence(task_id)
            if evidence is not None:
                evidence.state = state
                self._try_decide(task_id.utterance)

    def utterance_closed(self, task_id: TaskId, span: AudioSpan, close_reason: CloseReason,
                         speech_ended_at: float) -> None:
        with self._lock:
            record = self._get(task_id)
            if record is None:
                return
            if task_id in record.tasks:
                self._reject(f"fala {task_id} fechada duas vezes")
                return
            record.closed = True
            record.close_reason = close_reason.value
            record.speech_ended_at = speech_ended_at
            record.final_enqueued_at = self._clock()
            record.tasks[task_id] = TaskEvidence(task_id, span)

    def transcription(self, task_id: TaskId, result: Transcription | None, error: str | None,
                      normalized: str | None) -> None:
        with self._lock:
            evidence = self._evidence(task_id)
            if evidence is None:
                return
            if evidence.transcription is not None or evidence.asr_error is not None:
                self._reject(f"transcrição repetida para {task_id}")
                return
            evidence.transcription, evidence.asr_error, evidence.normalized = result, error, normalized
            self._records[task_id.utterance].final_transcribed_at = self._clock()
            if error is not None:
                evidence.state = TaskState.FAILED
            elif not normalized:
                # Sem texto: não há o que classificar (registrado como "pulado", não como erro).
                evidence.skipped.add(self._records[task_id.utterance].classifier)
                evidence.state = TaskState.NLP
            else:
                evidence.state = TaskState.NLP
            self._try_decide(task_id.utterance)

    def component_result(self, task_id: TaskId, component: str, result: Prediction | Reliability | None,
                         error: str | None = None) -> None:
        with self._lock:
            evidence = self._evidence(task_id)
            if evidence is None:
                return
            already = (component in evidence.predictions or component in evidence.component_errors
                       or (component == "fuzzy" and evidence.reliability is not None))
            if already:
                self._reject(f"resultado repetido de {component} para {task_id}")
                return
            if error is not None:
                evidence.component_errors[component] = error
            elif component == "fuzzy":
                evidence.reliability = result  # type: ignore[assignment]
            else:
                if not isinstance(result, Prediction) or result.model != component:
                    self._reject(f"resultado de {component} com modelo inconsistente para {task_id}")
                    return
                evidence.predictions[component] = result
            classifier = self._records[task_id.utterance].classifier
            if evidence.state is TaskState.NLP and not evidence.components_pending((classifier,)):
                evidence.state = TaskState.DONE
            self._try_decide(task_id.utterance)

    def check_timeouts(self) -> None:
        """Decide falas cuja final passou do prazo de segurança (chamado periodicamente)."""
        with self._lock:
            for utterance in list(self._records):
                self._try_decide(utterance)

    def pending(self) -> list[int]:
        with self._lock:
            return [u for u, r in self._records.items() if not r.decided]

    # ---------------------------------------------------------------- interno
    def _reject(self, message: str) -> None:
        self.rejected_events += 1
        log.error("evidência rejeitada: %s", message)

    def _get(self, task_id: TaskId) -> UtteranceRecord | None:
        record = self._records.get(task_id.utterance)
        if record is None:
            self._reject(f"{task_id} de fala desconhecida")
        return record

    def _evidence(self, task_id: TaskId) -> TaskEvidence | None:
        record = self._records.get(task_id.utterance)
        evidence = record.tasks.get(task_id) if record else None
        if evidence is None:
            if task_id.utterance in self._decided:
                self.late_events += 1
                log.warn("%s: resultado chegou depois da decisão da fala (ignorado)", task_id)
            else:
                self._reject(f"evidência para tarefa desconhecida {task_id}")
        return evidence

    def _try_decide(self, utterance: int) -> None:
        record = self._records.get(utterance)
        if record is None or record.decided or not record.closed:
            return
        final = record.tasks.get(TaskId(utterance))
        final_done = final is not None and (final.state in (TaskState.DONE, TaskState.FAILED))
        now = self._clock()
        nlp_late = (record.final_transcribed_at is not None
                    and now - record.final_transcribed_at > self._timeout)
        asr_late = (record.final_transcribed_at is None and record.final_enqueued_at is not None
                    and now - record.final_enqueued_at > self._asr_timeout)
        timed_out = nlp_late or asr_late
        if not timed_out and not final_done:
            return
        if timed_out and not final_done:
            what = "análises" if nlp_late else "transcrição final"
            limit = self._timeout if nlp_late else self._asr_timeout
            log.warn("U%04d: prazo de %.1f s das %s esgotado; decidindo com as evidências disponíveis",
                     utterance, limit, what)
        if final is None:
            return
        decision = decide(final, classifier=record.classifier, utterance=utterance,
                          close_reason=record.close_reason, speech_ended_at=record.speech_ended_at, now=now,
                          timed_out=timed_out)
        record.decided = True
        del self._records[utterance]
        self._decided.append(utterance)
        self._on_decision(decision)
