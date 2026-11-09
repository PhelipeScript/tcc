"""Segmentador: pre-roll, hangover, uma tarefa por fala, limite máximo e continuação."""

from __future__ import annotations

import numpy as np
import pytest

from ddsd.realtime.config import SegmentationConfig, VadConfig
from ddsd.realtime.domain import CloseReason, UtteranceClosed, UtteranceOpened
from ddsd.vad.segmenter import UtteranceSegmenter

CHUNK = 512
RATE = 16_000
VAD = VadConfig(threshold_on=0.5, threshold_off=0.35, min_speech_ms=250, min_silence_ms=700,
                pre_roll_ms=300, post_roll_ms=300)
SEG = SegmentationConfig(max_utterance_s=30.0)


def run(probs: list[float], seg: SegmentationConfig = SEG, vad: VadConfig = VAD):
    """Alimenta blocos cujo valor = índice da amostra absoluta, para checar o áudio bit a bit."""
    segmenter = UtteranceSegmenter(vad, seg)
    events = []
    for i, p in enumerate(probs):
        chunk = np.arange(i * CHUNK, (i + 1) * CHUNK, dtype=np.float32)
        events += segmenter.feed(chunk, p, i * CHUNK, captured_at=float(i), now=float(i))
    return segmenter, events


def chunks_for(seconds: float) -> int:
    return round(seconds * RATE / CHUNK)


def test_utterance_has_preroll_postroll_and_exact_audio() -> None:
    probs = [0.0] * 30 + [0.9] * chunks_for(2.0) + [0.0] * 40
    _, events = run(probs)
    opened = [e for e in events if isinstance(e, UtteranceOpened)]
    closed = [e for e in events if isinstance(e, UtteranceClosed)]
    assert len(opened) == 1 and len(closed) == 1 and len(events) == 2
    task = closed[0].task
    speech_start = 30 * CHUNK
    pre_roll = round(0.3 * RATE)
    assert task.span.start == speech_start - pre_roll
    speech_end = (30 + chunks_for(2.0)) * CHUNK
    assert task.span.end == speech_end + round(0.3 * RATE)  # pós-roll limitado
    # Áudio bit a bit igual ao fluxo de entrada (valor = índice da amostra).
    assert np.array_equal(task.audio, np.arange(task.span.start, task.span.end, dtype=np.float32))
    assert task.close_reason is CloseReason.END_OF_SPEECH
    assert str(task.task_id) == "U0001-F"


def test_long_utterance_produces_only_the_full_utterance_task() -> None:
    probs = [0.0] * 10 + [0.9] * chunks_for(14.0) + [0.0] * 40
    _, events = run(probs)
    assert [type(e) for e in events] == [UtteranceOpened, UtteranceClosed]
    task = events[1].task
    assert task.span.samples >= 14 * RATE
    assert np.array_equal(task.audio, np.arange(task.span.start, task.span.end, dtype=np.float32))


def test_short_pause_does_not_split_but_long_pause_does() -> None:
    pause_short = [0.0] * chunks_for(0.5)
    pause_long = [0.0] * chunks_for(1.2)
    probs = [0.9] * 30 + pause_short + [0.9] * 30 + pause_long + [0.9] * 30 + [0.0] * 40
    _, events = run(probs)
    closed = [e for e in events if isinstance(e, UtteranceClosed)]
    assert len(closed) == 2


def test_spurious_blip_is_ignored() -> None:
    probs = [0.0] * 20 + [0.9] * 3 + [0.0] * 60  # 96 ms < min_speech de 250 ms
    _, events = run(probs)
    assert events == []


def test_max_duration_closes_and_continues_without_losing_audio() -> None:
    seg = SegmentationConfig(max_utterance_s=6.0)
    probs = [0.9] * chunks_for(14.0) + [0.0] * 40
    _, events = run(probs, seg=seg)
    finals = [e.task for e in events if isinstance(e, UtteranceClosed)]
    assert [t.close_reason for t in finals[:-1]] == [CloseReason.MAX_DURATION] * (len(finals) - 1)
    for a, b in zip(finals, finals[1:], strict=False):
        assert b.span.start == a.span.end  # continuação começa na amostra seguinte
        assert b.continuation_of == a.task_id.utterance
    opened = [e for e in events if isinstance(e, UtteranceOpened)]
    assert [o.utterance for o in opened] == list(range(1, len(finals) + 1))


def test_input_gap_closes_utterance_with_error() -> None:
    segmenter = UtteranceSegmenter(VAD, SEG)
    events = []
    for i in range(20):
        chunk = np.zeros(CHUNK, dtype=np.float32)
        events += segmenter.feed(chunk, 0.9, i * CHUNK, captured_at=0.0, now=0.0)
    events += segmenter.feed(np.zeros(CHUNK, np.float32), 0.9, 25 * CHUNK, captured_at=0.0, now=0.0)
    closed = [e for e in events if isinstance(e, UtteranceClosed)]
    assert closed and closed[0].task.close_reason is CloseReason.ERROR


def test_shutdown_close_flushes_open_utterance() -> None:
    segmenter, events = run([0.9] * 40)
    assert not [e for e in events if isinstance(e, UtteranceClosed)]
    final = segmenter.close(CloseReason.SHUTDOWN, now=1.0)
    assert len(final) == 1 and final[0].task.close_reason is CloseReason.SHUTDOWN


def test_task_audio_is_read_only() -> None:
    _, events = run([0.9] * 30 + [0.0] * 40)
    task = next(e for e in events if isinstance(e, UtteranceClosed)).task
    with pytest.raises(ValueError):
        task.audio[0] = 1.0
