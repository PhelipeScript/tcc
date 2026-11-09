"""Controles do painel: pausa do microfone, troca de classificador e teclas.

Mocks permitidos (teste unitário): VAD por energia, captura e classificadores falsos. O
pseudoterminal é real.
"""

from __future__ import annotations

import os
import pty
import termios
import time
from types import SimpleNamespace

import numpy as np
import pytest

from ddsd.audio.ring_buffer import AudioRingBuffer
from ddsd.observability.status import StatusBoard
from ddsd.realtime import frontend as frontend_module
from ddsd.realtime.config import load_pipeline_config
from ddsd.realtime.domain import CloseReason, UtteranceClosed, UtteranceOpened
from ddsd.realtime.keys import KeyReader, handle_key, keys_of
from test_pipeline_stress import FakeClassifier, make_pipeline, start_without_audio, utterance_events


def wait_for(condition, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


class EnergyVad:
    """VAD falso: fala quando o bloco tem energia (o Silero real fica nos testes de integração)."""

    def probability(self, chunk: np.ndarray) -> float:
        return 0.9 if float(np.sqrt(np.mean(np.square(chunk)))) > 0.05 else 0.05

    def reset(self) -> None:
        pass


@pytest.fixture
def frontend(monkeypatch):
    monkeypatch.setattr(frontend_module, "SileroVad", EnergyVad)
    monkeypatch.setattr(frontend_module, "STALL_SECONDS", 0.2)
    cfg = load_pipeline_config()
    ring = AudioRingBuffer(48_000 * 10)
    events, stalls, status = [], [], StatusBoard()
    fe = frontend_module.AudioFrontend(cfg, ring, lambda: 48_000, events.extend, status, lambda: stalls.append(1))
    fe.start()
    yield SimpleNamespace(fe=fe, ring=ring, events=events, stalls=stalls, status=status)
    fe.stop()
    fe.join(5)


def tone(seconds: float, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(48_000 * seconds)) / 48_000
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def feed(ring: AudioRingBuffer, signal: np.ndarray) -> None:
    for i in range(0, len(signal), 480):
        ring.write(signal[i:i + 480], time.monotonic())


def test_pause_closes_the_open_utterance_and_never_reopens_the_microphone(frontend) -> None:
    feed(frontend.ring, tone(1.2))
    assert wait_for(lambda: any(isinstance(e, UtteranceOpened) for e in frontend.events))
    frontend.fe.set_paused(True)
    assert wait_for(lambda: any(isinstance(e, UtteranceClosed) for e in frontend.events))
    closed = next(e for e in frontend.events if isinstance(e, UtteranceClosed))
    assert closed.task.close_reason is CloseReason.PAUSE  # a fala aberta é decidida, não perdida
    feed(frontend.ring, tone(1.0))  # áudio que ainda estava no buffer: pausado, não abre fala nova
    time.sleep(0.3)
    assert sum(isinstance(e, UtteranceOpened) for e in frontend.events) == 1
    time.sleep(0.6)  # sem áudio por mais que o prazo de "travamento": pausado, ninguém reabre o microfone
    assert frontend.stalls == []
    frontend.fe.set_paused(False)
    assert wait_for(lambda: frontend.stalls, timeout=2.0)  # retomado e sem áudio: volta a vigiar


# --------------------------------------------------------------- pipeline
class FakeCapture:
    def __init__(self) -> None:
        self.samplerate, self.device = 48_000, SimpleNamespace(name="microfone falso")
        self.running, self.overflows, self.callbacks = True, 0, 0

    @property
    def active(self) -> bool:
        return self.running

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False


class FakeFrontend:
    def __init__(self) -> None:
        self.paused, self.error = False, None

    def set_paused(self, paused: bool) -> None:
        self.paused = paused

    def stop(self) -> None:
        pass

    def join(self, timeout: float | None = None) -> None:
        pass


def test_pause_and_resume_the_microphone_from_the_panel() -> None:
    pipeline = make_pipeline(asr_delay=0.01)
    start_without_audio(pipeline)
    pipeline.capture, pipeline.frontend = FakeCapture(), FakeFrontend()
    try:
        pipeline.toggle_capture()
        assert pipeline.capture_paused and not pipeline.capture.running and pipeline.frontend.paused
        assert pipeline.activity() == "pausado" and pipeline.status.get("captura", "estado") == "pausada"
        pipeline.toggle_capture()
        assert not pipeline.capture_paused and pipeline.capture.running and not pipeline.frontend.paused
        assert pipeline.activity() == "ocioso" and pipeline.status.get("captura", "estado") == "ativa"
    finally:
        pipeline.frontend = None
        pipeline.stop()


def test_classifier_switch_loads_once_and_applies_from_the_next_utterance() -> None:
    FakeClassifier.built.clear()
    pipeline = make_pipeline(asr_delay=0.01)  # BERTimbau ativo
    start_without_audio(pipeline)
    try:
        opened1, closed1 = utterance_events(1, start=0)
        pipeline._on_events([opened1])  # noqa: SLF001 - U1 abre com o BERTimbau
        pipeline.set_classifier("debertinha")
        assert pipeline.status.get("pipeline", "carregando") == "debertinha" or pipeline.classifier_name == "debertinha"
        assert wait_for(lambda: pipeline.classifier_name == "debertinha")
        assert pipeline.status.get("pipeline", "carregando") is None
        pipeline._on_events([closed1])  # noqa: SLF001 - fecha depois da troca: continua com o da abertura
        for event in utterance_events(2, start=1_000_000):
            pipeline._on_events([event])  # noqa: SLF001 - U2 já abre com a DeBERTinha
        assert wait_for(lambda: len(pipeline.decisions) == 2)
        pipeline.set_classifier("bertimbau")  # já carregado: troca na hora, sem carregar de novo
        assert pipeline.classifier_name == "bertimbau"
    finally:
        pipeline.stop()
    by = {d.utterance: d for d in pipeline.decisions}
    assert (by[1].classifier, by[2].classifier) == ("bertimbau", "debertinha")
    assert set(by[1].votes) == {"bertimbau"} and set(by[2].votes) == {"debertinha"}
    assert FakeClassifier.built == ["bertimbau", "debertinha"]
    views = {u.utterance: u for u in pipeline.status.snapshot().utterances}
    assert (views["U0001"].classifier, views["U0002"].classifier) == ("bertimbau", "debertinha")
    with pytest.raises(ValueError):
        pipeline.set_classifier("gpt")


# ----------------------------------------------------------------- teclas
class Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.classifier_name = "debertinha"

    def toggle_capture(self) -> None:
        self.calls.append(("microfone",))

    def set_classifier(self, name: str) -> None:
        self.calls.append(("classificador", name))


def test_key_map() -> None:
    p = Recorder()
    assert [handle_key(p, k) for k in ("M", " ", "1", "2", "3", "c", "q", "x")] == [
        "microfone", "microfone", "bertimbau", "albertina", "debertinha", "proximo_classificador", "sair", None]
    assert p.calls == [("microfone",), ("microfone",), ("classificador", "bertimbau"),
                       ("classificador", "albertina"), ("classificador", "debertinha"),
                       ("classificador", "bertimbau")]  # "c" depois da DeBERTinha volta ao BERTimbau


def test_next_key_with_a_tfidf_baseline_active_goes_to_the_first_transformer() -> None:
    p = Recorder()
    p.classifier_name = "baseline_svm"
    assert handle_key(p, "c") == "proximo_classificador"
    assert p.calls == [("classificador", "bertimbau")]
    assert keys_of("r\x1b[A1\x1bOPq") == ["r", "1", "q"]  # setas e F1 não viram teclas


def test_key_reader_uses_cbreak_and_restores_the_terminal() -> None:
    master, slave = pty.openpty()
    try:
        before = termios.tcgetattr(slave)
        got: list[str] = []
        reader = KeyReader(got.append, fd=slave)
        assert reader.start()
        assert not termios.tcgetattr(slave)[3] & (termios.ICANON | termios.ECHO)  # sem Enter e sem eco
        os.write(master, b"r1\x1b[Aq")
        assert wait_for(lambda: got == ["r", "1", "q"])
        reader.stop()
        after = termios.tcgetattr(slave)
        # PENDIN é ligado pelo próprio kernel ao voltar ao modo canônico; o resto tem de ser idêntico.
        after[3] &= ~termios.PENDIN
        assert after == before
    finally:
        os.close(master)
        os.close(slave)


def test_key_reader_is_off_without_a_terminal(tmp_path) -> None:
    with (tmp_path / "entrada").open("w+") as handle:
        assert not KeyReader(lambda k: None, fd=handle.fileno()).start()
