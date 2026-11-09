"""Gravação do conjunto de avaliação: stream contínuo, detecção de perda e fluxo de regravação."""

from __future__ import annotations

import json

import numpy as np
import soundfile as sf

from ddsd.evaluation import recording
from ddsd.evaluation.recording import MAX_LOSS_S, Recorder, Scenario, Take


class FakeClock:
    def __init__(self, *times: float) -> None:
        self.times = list(times)

    def __call__(self) -> float:
        return self.times.pop(0)


def block(n: int, value: float = 0.1) -> np.ndarray:
    return np.full((n, 1), value, dtype=np.float32)


def test_recorder_keeps_only_audio_between_start_and_stop_and_measures_loss() -> None:
    rec = Recorder(device_index=0, samplerate=1000, clock=FakeClock(100.0, 103.0))
    rec._callback(block(100), 100, None, None)  # antes de começar: descartado
    rec.start()
    for _ in range(10):  # 1 s de áudio em 3 s de parede: o dispositivo deixou de entregar 2 s
        rec._callback(block(100), 100, None, None)
    take = rec.stop()
    rec._callback(block(100), 100, None, None)  # depois de parar: descartado
    assert take.received_s == 1.0 and take.elapsed_s == 3.0
    assert abs(take.lost_s - 2.0) < 1e-9 and take.lost_s > MAX_LOSS_S
    assert take.status_flags == 0 and len(take.audio) == 1000


def test_recorder_counts_portaudio_flags_only_while_recording() -> None:
    rec = Recorder(device_index=0, samplerate=1000, clock=FakeClock(0.0, 0.2))
    rec._callback(block(100), 100, None, "input overflow")
    rec.start()
    rec._callback(block(100), 100, None, "input overflow")
    rec._callback(block(100), 100, None, None)
    take = rec.stop()
    assert take.status_flags == 1
    assert take.lost_s == 0.0  # recebeu 0,2 s em 0,2 s


class FakePrompt:
    """Respostas roteirizadas; `None` aceita o padrão oferecido pela sessão."""

    def __init__(self, answers: list[str | None]) -> None:
        self.answers = answers
        self.defaults: list[str | None] = []

    def ask(self, question: str, choices=None, default=None):  # noqa: ANN001
        self.defaults.append(default)
        answer = self.answers.pop(0)
        return default if answer is None else answer


class FakeConsole:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def print(self, text: str = "") -> None:
        self.lines.append(str(text))


class FakeRecorder:
    def __init__(self, takes: list[Take]) -> None:
        self.takes = takes
        self.started = 0

    def start(self) -> None:
        self.started += 1

    def stop(self) -> Take:
        return self.takes.pop(0)


def test_session_suggests_rerecording_on_loss_and_saves_only_the_good_take(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(recording, "DATA_DIR", tmp_path)
    monkeypatch.setattr(recording, "EVAL_DIR", tmp_path / "eval" / "gravacoes")
    rate = 1000
    lossy = Take(np.full(200, 0.2, dtype=np.float32), elapsed_s=2.0, samplerate=rate, status_flags=0)
    good = Take(np.full(1000, 0.2, dtype=np.float32), elapsed_s=1.0, samplerate=rate, status_flags=0)
    recorder = FakeRecorder([lossy, good])
    # Por take: Enter para gravar, Enter para parar, escolha (None = aceita o padrão da sessão).
    prompt = FakePrompt(["", "", None, "", "", None])
    console = FakeConsole()
    device = type("Device", (), {"name": "MacBook Air Microphone", "samplerate": rate})()
    scenario = Scenario("quieto_interno", "Sala quieta", "MacBook", "")
    manifest = tmp_path / "eval" / "gravacoes" / "manifest.jsonl"
    items = [("P00", {"label": "DD", "text_raw": "abre o navegador", "bin": "3"})]

    result = recording._record_scenario(console, prompt, recorder, scenario, device, items, set(), manifest)

    assert result == "ok" and recorder.started == 2
    assert prompt.defaults[2] == "r" and prompt.defaults[5] == "o"  # com perda, o padrão é regravar
    assert sum("perda de áudio" in line for line in console.lines) == 1
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["item"] == "P00" and rows[0]["lost_s"] == 0.0
    assert rows[0]["elapsed_s"] == 1.0 and rows[0]["duration_s"] == 1.0
    assert sf.info(str(tmp_path / rows[0]["file"])).frames == 1000


def test_session_exit_stops_without_writing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(recording, "DATA_DIR", tmp_path)
    monkeypatch.setattr(recording, "EVAL_DIR", tmp_path / "eval" / "gravacoes")
    take = Take(np.full(500, 0.2, dtype=np.float32), elapsed_s=0.5, samplerate=1000, status_flags=0)
    device = type("Device", (), {"name": "MacBook Air Microphone", "samplerate": 1000})()
    manifest = tmp_path / "eval" / "gravacoes" / "manifest.jsonl"
    result = recording._record_scenario(FakeConsole(), FakePrompt(["", "", "s"]), FakeRecorder([take]),
                                        Scenario("quieto_interno", "Sala quieta", "MacBook", ""), device,
                                        [("P00", {"label": "NDD", "text_raw": "cadê meu café", "bin": "3"})],
                                        set(), manifest)
    assert result == "sair" and not manifest.exists()


def test_conversation_script_is_read_in_order_as_ndd_lines_with_reference(tmp_path) -> None:
    script = tmp_path / "roteiro.txt"
    script.write_text("# comentário\nBom dia, dormiu bem?\n\n  Me passa o carregador.  \n", encoding="utf-8")
    prompts = recording.conversation_prompts(script)
    assert [(p["prompt_id"], p["text_raw"], p["label"]) for p in prompts] == [
        ("C01", "Bom dia, dormiu bem?", "NDD"), ("C02", "Me passa o carregador.", "NDD")]
    assert prompts[0]["text"] == "bom dia dormiu bem"
    real = recording.conversation_prompts()
    assert len(real) >= 20 and all(p["label"] == "NDD" and p["text"] for p in real)
    assert recording.CONVERSATION_SCENARIO in {s.key for s in recording.SCENARIOS}
