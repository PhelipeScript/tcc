"""Fila de tarefas (FIFO, fechamento, concorrência sem perda) e lógica fuzzy de confiabilidade."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import pytest

from ddsd.asr.types import WhisperSegment
from ddsd.fuzzy.reliability import ReliabilityInputs, evaluate, reliability_of
from ddsd.realtime.job_queue import JobQueue, QueueClosed


@dataclass(frozen=True)
class Job:
    utterance: int
    seq: int = 0


def test_queue_is_fifo_and_counts() -> None:
    q: JobQueue[Job] = JobQueue("teste")
    for job in (Job(1), Job(2), Job(3)):
        q.put(job)
    assert q.stats().pending == 3 and q.stats().enqueued == 3
    assert [q.get(0) for _ in range(3)] == [Job(1), Job(2), Job(3)]
    assert q.get(timeout=0.01) is None
    assert q.stats().pending == 0 and q.stats().enqueued == 3


def test_queue_never_drops_under_burst() -> None:
    q: JobQueue[Job] = JobQueue("teste")
    for u in range(500):
        q.put(Job(u))
    assert q.stats().pending == 500
    assert [q.get(0).utterance for _ in range(500)] == list(range(500))


def test_close_wakes_consumers_and_drains_what_is_left() -> None:
    q: JobQueue[Job] = JobQueue("teste")
    errors = []

    def consumer() -> None:
        try:
            q.get(timeout=5)
        except QueueClosed:
            errors.append("closed")

    thread = threading.Thread(target=consumer)
    thread.start()
    q.close()
    thread.join(2)
    assert errors == ["closed"]
    with pytest.raises(QueueClosed):
        q.put(Job(1))
    pending: JobQueue[Job] = JobQueue("teste")
    pending.put(Job(7))
    pending.close()
    assert pending.get(0) == Job(7)  # o que já estava na fila ainda é entregue
    with pytest.raises(QueueClosed):
        pending.get(0)


def test_concurrent_producers_lose_nothing() -> None:
    q: JobQueue[Job] = JobQueue("teste")
    consumed: list[Job] = []
    done = threading.Event()

    def consumer() -> None:
        while True:
            try:
                job = q.get(timeout=0.5)
            except QueueClosed:
                return
            if job is not None:
                consumed.append(job)
            elif done.is_set():
                return

    workers = [threading.Thread(target=lambda u=u: [q.put(Job(u, i)) for i in range(100)]) for u in range(8)]
    reader = threading.Thread(target=consumer)
    reader.start()
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    done.set()
    reader.join(5)
    q.close()
    assert len(consumed) == 800 and q.stats().enqueued == 800
    for u in range(8):  # a ordem de cada produtor é preservada
        assert [j.seq for j in consumed if j.utterance == u] == list(range(100))


def seg(avg_logprob: float, no_speech: float, ratio: float, temp: float = 0.0, dur: float = 2.0) -> WhisperSegment:
    return WhisperSegment(0.0, dur, "x", (1,), avg_logprob, ratio, no_speech, temp, 0)


def test_fuzzy_clean_speech_is_high() -> None:
    result = reliability_of([seg(-0.1, 0.02, 1.3)])
    assert result.label == "alta" and result.score > 0.7


def test_fuzzy_hallucination_cases_are_low() -> None:
    assert reliability_of([seg(-0.2, 0.9, 1.3)]).label == "baixa"  # silêncio
    assert reliability_of([seg(-0.2, 0.05, 3.0)]).label == "baixa"  # repetição
    assert reliability_of([seg(-1.6, 0.05, 1.3)]).label == "baixa"  # decodificação fraca
    assert reliability_of([]).score == 0.0


def test_fuzzy_is_monotonic_in_each_input() -> None:
    base = dict(avg_logprob=-0.3, no_speech_prob=0.1, compression_ratio=1.5, temperature=0.0)
    previous = 2.0
    for value in (-0.05, -0.3, -0.6, -0.9, -1.2, -2.0):  # pior confiança média → r não sobe
        score = evaluate(ReliabilityInputs(**{**base, "avg_logprob": value})).score
        assert score <= previous + 1e-9
        previous = score
    previous = 2.0
    for value in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):  # mais "sem fala" → r não sobe
        score = evaluate(ReliabilityInputs(**{**base, "no_speech_prob": value})).score
        assert score <= previous + 1e-9
        previous = score


def test_fuzzy_weights_segments_by_duration() -> None:
    long_good = seg(-0.1, 0.02, 1.3, dur=9.0)
    short_bad = seg(-2.0, 0.02, 1.3, dur=0.5)
    assert reliability_of([long_good, short_bad]).score > reliability_of([seg(-2.0, 0.02, 1.3)]).score
