"""Estresse das políticas do pipeline com ASR e classificadores falsos e lentos (mocks permitidos em teste).

Alimenta o `Pipeline` real (filas, workers, agregador, decisor) com eventos do segmentador e
verifica, sob sobrecarga, que:
- nenhuma fala é perdida e toda fala recebe decisão;
- os resultados de falas diferentes nunca se misturam.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from ddsd.asr.types import Transcription, WhisperSegment
from ddsd.classifiers.model import Prediction
from ddsd.realtime.config import PipelineConfigError, load_pipeline_config
from ddsd.realtime.domain import AudioSpan, AudioTask, CloseReason, TaskId, UtteranceClosed, UtteranceOpened
from ddsd.realtime.pipeline import Pipeline


class SlowBackend:
    name = "falso"
    model_id = "falso"

    def __init__(self, delay: float) -> None:
        self.delay = delay

    def warm_up(self) -> float:
        return 0.0

    def transcribe(self, audio: np.ndarray) -> Transcription:
        time.sleep(self.delay)
        # O texto codifica o tamanho do áudio, para checar que cada resultado é da sua própria tarefa.
        text = f" fala com {len(audio)} amostras"
        seg = WhisperSegment(0.0, len(audio) / 16_000, text, (1,), -0.2, 1.2, 0.05, 0.0, 0)
        return Transcription(text, "pt", (seg,), "falso", "falso", len(audio) / 16_000, self.delay)


class FakeClassifier:
    built: list[str] = []  # nomes de todos os classificadores construídos (só o ativo deve ser carregado)

    def __init__(self, name: str) -> None:
        self.name = name
        FakeClassifier.built.append(name)

    def warm_up(self) -> None:
        pass

    def predict(self, text: str) -> Prediction:
        time.sleep(0.002)
        label = "DD" if int(text.split()[2]) % 2 == 0 else "NDD"
        return Prediction(self.name, 0.9 if label == "DD" else 0.1, 0.5, label, 2.0, 4, False)


def make_pipeline(asr_delay: float, recorder=None, **overrides) -> Pipeline:
    cfg = load_pipeline_config(**overrides)
    pipeline = Pipeline(cfg, recorder=recorder)
    pipeline._make_backend = lambda options: SlowBackend(asr_delay)  # noqa: SLF001 - injeção só no teste
    pipeline._make_classifier = lambda name: FakeClassifier(name)  # noqa: SLF001
    pipeline.asr_worker._factory = lambda: SlowBackend(asr_delay)  # noqa: SLF001
    return pipeline


def start_without_audio(pipeline: Pipeline) -> None:
    workers = pipeline.workers
    for w in workers:
        w.start()
    for w in workers:
        w.ready.wait()
    pipeline._monitor.start()  # noqa: SLF001


def utterance_events(u: int, start: int) -> list:
    """Abertura e fechamento de uma fala com áudio de comprimento distinto por fala."""
    final_len = 16_000 * (2 + u)
    final = AudioTask.create(TaskId(u), AudioSpan(start, start + final_len), np.zeros(final_len, np.float32),
                             time.monotonic(), time.monotonic(), close_reason=CloseReason.END_OF_SPEECH)
    return [UtteranceOpened(u, start, time.monotonic()), UtteranceClosed(final, time.monotonic())]


def test_overload_queues_every_utterance_and_decides_each_with_its_own_audio() -> None:
    pipeline = make_pipeline(asr_delay=0.15)
    start_without_audio(pipeline)
    try:
        position = 0
        for u in range(1, 9):  # 8 falas fechando juntas: o ASR fica com fila, nada é descartado
            for event in utterance_events(u, start=position):
                pipeline._on_events([event])  # noqa: SLF001
            position += 1_000_000
        deadline = time.monotonic() + 30
        while len(pipeline.decisions) < 8 and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        pipeline.stop()
    decisions = {d.utterance: d for d in pipeline.decisions}
    assert sorted(decisions) == list(range(1, 9))  # toda fala decidida
    for u, d in decisions.items():
        assert d.text_raw == f" fala com {16_000 * (2 + u)} amostras"  # o texto é o desta fala
        assert d.label == ("DD" if (16_000 * (2 + u)) % 2 == 0 else "NDD")
        assert [str(e.task_id) for e in d.evidence] == [f"U{u:04d}-F"]  # uma única tarefa por fala
    assert pipeline.aggregator.rejected_events == 0
    stats = pipeline.asr_queue.stats()
    assert stats.enqueued == 8 and stats.pending == 0


def test_only_the_active_classifier_runs_and_the_panel_follows_the_utterance() -> None:
    FakeClassifier.built.clear()
    pipeline = make_pipeline(asr_delay=0.01, classifiers__active="albertina")
    start_without_audio(pipeline)
    try:
        assert FakeClassifier.built == ["albertina"]  # as outras execuções nem são carregadas
        assert set(pipeline.nlp_queues) == {"fuzzy", "albertina"}
        for event in utterance_events(1, start=0):
            pipeline._on_events([event])  # noqa: SLF001
        deadline = time.monotonic() + 10
        while not pipeline.decisions and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        pipeline.stop()
    d = pipeline.decisions[0]
    assert d.classifier == "albertina" and set(d.votes) == {"albertina"} and d.reason == "classificador"
    view = pipeline.status.snapshot().utterances[-1]
    assert (view.utterance, view.state, view.label) == ("U0001", "decidida", d.label)
    assert view.final_text == d.text_raw and view.final_normalized == d.normalized
    assert view.duration_s == d.span.duration_s


def test_config_rejects_unknown_active_classifier() -> None:
    with pytest.raises(PipelineConfigError, match="classifiers.active"):
        load_pipeline_config(classifiers__active="gpt")


def test_session_recording_keeps_each_utterance_audio_with_its_decision(tmp_path) -> None:
    import json

    import soundfile as sf

    from ddsd.realtime.session import EVIDENCE_FILE, SESSION_FILE, SessionRecorder

    recorder = SessionRecorder(tmp_path / "sessao", {"comando": "teste"})
    pipeline = make_pipeline(asr_delay=0.02, recorder=recorder)
    start_without_audio(pipeline)
    try:
        position = 0
        for u in range(1, 5):
            events = utterance_events(u, start=position)
            closed = events[-1]
            level = np.full(closed.task.span.samples, u / 10, np.float32)  # conteúdo distinto por fala
            final = AudioTask.create(closed.task.task_id, closed.task.span, level, time.monotonic(),
                                     time.monotonic(), close_reason=CloseReason.END_OF_SPEECH)
            for event in [*events[:-1], UtteranceClosed(final, time.monotonic())]:
                pipeline._on_events([event])  # noqa: SLF001
            position += 1_000_000
        deadline = time.monotonic() + 20
        while len(pipeline.decisions) < 4 and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        pipeline.stop()
        summary = recorder.close()
    assert summary["falas_gravadas"] == 4 and summary["decisoes_gravadas"] == 4 and summary["erros"] == 0
    lines = [json.loads(x) for x in (tmp_path / "sessao" / EVIDENCE_FILE).read_text(encoding="utf-8").splitlines()]
    decided = {d.utterance_label: d for d in pipeline.decisions}
    assert sorted(r["utterance"] for r in lines) == ["U0001", "U0002", "U0003", "U0004"]
    for record in lines:
        u = int(record["utterance"][1:])
        assert record["audio"] == f"U{u:04d}.wav" and record["label"] == decided[record["utterance"]].label
        audio, rate = sf.read(tmp_path / "sessao" / record["audio"], dtype="float32")
        assert rate == 16_000 and len(audio) == 16_000 * (2 + u)  # a fala inteira, sem cortes
        assert np.allclose(audio, u / 10, atol=1 / 16_000)  # o áudio é o desta fala (PCM 16 bits)
    session = json.loads((tmp_path / "sessao" / SESSION_FILE).read_text(encoding="utf-8"))
    assert session["metadados"] == {"comando": "teste"} and session["resumo"]["pendentes_no_fim"] == 0


def test_config_accepts_the_tfidf_baselines_and_validates_the_whisper_seed() -> None:
    cfg = load_pipeline_config(classifiers__active="baseline_svm")
    assert cfg.classifiers.active_run.name == "svm.joblib"
    assert cfg.classifiers.run_for("baseline_logreg").name == "regressao_logistica.joblib"
    assert set(cfg.classifiers.runs()) == {"bertimbau", "albertina", "debertinha"}  # o painel troca só estes
    assert isinstance(cfg.asr.seed, int) and cfg.asr.seed >= 0
    with pytest.raises(PipelineConfigError, match="asr.seed"):
        load_pipeline_config(asr__seed=-1)


def test_pipeline_builds_a_tfidf_baseline_as_its_classifier(monkeypatch) -> None:
    import ddsd.classifiers.hub as hub
    import ddsd.evaluation.baselines as baselines

    built, ensured = [], []

    class FakeBaseline:
        def __init__(self, run_dir) -> None:
            built.append(run_dir.name)

    monkeypatch.setattr(baselines, "BaselineClassifier", FakeBaseline)
    monkeypatch.setattr(hub, "ensure", lambda path, repos: ensured.append((path.name, repos)) or path)  # sem rede
    cfg = load_pipeline_config(classifiers__active="baseline_logreg")
    pipeline = Pipeline(cfg)
    assert isinstance(pipeline._make_classifier("baseline_logreg"), FakeBaseline)  # noqa: SLF001
    assert built == ["regressao_logistica.joblib"]
    assert ensured == [("regressao_logistica.joblib", cfg.hub.repos)]
    with pytest.raises(ValueError):
        pipeline.set_classifier("gpt")


def test_mlx_backend_fixes_the_random_seed_before_any_decoding(monkeypatch) -> None:
    import mlx.core as mx

    from ddsd.asr.backends import MlxWhisperBackend
    from ddsd.asr.types import DecodingOptions

    seeds = []
    monkeypatch.setattr(mx.random, "seed", lambda value: seeds.append(value))
    MlxWhisperBackend("modelo", DecodingOptions(), seed=7)
    MlxWhisperBackend("modelo", DecodingOptions())
    assert seeds == [7]
