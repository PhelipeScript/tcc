"""Integração com modelos e hardware reais (sem mocks).

Rodar com: `uv run pytest tests/integration -m "models or audio_hw" -v`
- `models`: exige o Whisper em cache e classificadores treinados (usa os de fumaça, se existirem);
- `audio_hw`: exige microfone (e alto-falante, para o loopback).
"""

from __future__ import annotations

import dataclasses
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ddsd.common.paths import MODELS_DIR, PROJECT_ROOT

SMOKE = {name: MODELS_DIR / "smoke" / run for name, run in
         (("bertimbau", "bertimbau-auto"), ("albertina", "albertina-fp32"), ("debertinha", "debertinha-auto"))}


def say(tmp_path: Path, name: str, text: str, rate: int = 16_000) -> Path:
    path = tmp_path / f"{name}.wav"
    subprocess.run(["say", "-v", "Luciana", "-o", str(path), f"--data-format=LEI16@{rate}", text], check=True)
    return path


@pytest.fixture(scope="module")
def speech_clip(tmp_path_factory) -> tuple[np.ndarray, str]:
    text = "abre o navegador e pesquisa receitas de bolo de chocolate"
    path = say(tmp_path_factory.mktemp("tts"), "fala", text)
    audio, rate = sf.read(str(path), dtype="float32")
    assert rate == 16_000
    return audio, text


@pytest.mark.models
@pytest.mark.parametrize("backend_name", ["mlx", "faster-whisper"])
def test_whisper_backends_share_metadata_structure(backend_name: str, speech_clip) -> None:
    from ddsd.asr.backends import create_backend
    from ddsd.asr.types import DecodingOptions, WhisperSegment
    from ddsd.evaluation.asr_impact import wer

    audio, reference = speech_clip
    backend = create_backend(backend_name, mlx_model="mlx-community/whisper-large-v3-turbo",
                             faster_whisper_model="mobiuslabsgmbh/faster-whisper-large-v3-turbo",
                             compute_type="float32", options=DecodingOptions())
    result = backend.transcribe(audio)
    assert result.language == "pt"
    assert result.segments and all(isinstance(s, WhisperSegment) for s in result.segments)
    fields = {f.name for f in dataclasses.fields(WhisperSegment)}
    assert fields == {"start", "end", "text", "tokens", "avg_logprob", "compression_ratio", "no_speech_prob",
                      "temperature", "seek"}
    assert result.text_raw == "".join(s.text for s in result.segments)  # RAW sem alteração
    assert wer(reference, result.text_raw) < 0.3
    assert result.rtf > 0


def _smoke_config(tmp_path: Path) -> Path:
    if not all((p / "run_manifest.json").exists() for p in SMOKE.values()):
        pytest.skip("classificadores de fumaça indisponíveis (rode `ddsd train <modelo> --smoke`)")
    text = (PROJECT_ROOT / "config" / "pipeline.toml").read_text(encoding="utf-8")
    for name, path in SMOKE.items():
        text = "\n".join(f'{name} = "{path}"' if line.startswith(f"{name} = ") else line for line in text.splitlines())
    config = tmp_path / "pipeline.toml"
    config.write_text(text, encoding="utf-8")
    return config


@pytest.mark.models
def test_pipeline_replay_end_to_end(tmp_path: Path) -> None:
    from ddsd.realtime.config import load_pipeline_config
    from ddsd.realtime.pipeline import Pipeline, load_audio, utterances_for_file

    cfg = load_pipeline_config(_smoke_config(tmp_path))
    clips = [say(tmp_path, "dd", "pausa o vídeo", 48_000),
             say(tmp_path, "ndd", "você viu aquele vídeo novo que a marina mandou ontem no grupo da família", 48_000)]
    pipeline = Pipeline(cfg)
    pipeline.start(file_rate=48_000)
    per_file = []
    try:
        for clip in clips:
            audio, _ = load_audio(clip)
            start16, end16 = pipeline.play(audio, trailing_silence_s=1.5)
            assert pipeline.wait_idle(timeout_s=90)
            per_file.append(utterances_for_file(pipeline.decisions, start16, end16))
    finally:
        pipeline.stop()
    assert [len(d) for d in per_file] == [1, 1]
    for decisions in per_file:
        d = decisions[0]
        assert d.label in {"DD", "NDD"} and d.text_raw.strip()
        assert d.classifier == cfg.classifiers.active and set(d.votes) <= {cfg.classifiers.active}
        assert all(e.task_id.utterance == d.utterance for e in d.evidence)
    assert pipeline.aggregator.rejected_events == 0


@pytest.mark.audio_hw
def test_microphone_capture_continuity() -> None:
    from ddsd.audio.capture import AudioCapture, resolve_device
    from ddsd.audio.ring_buffer import AudioRingBuffer

    device = resolve_device("")
    ring = AudioRingBuffer(device.samplerate * 10)
    capture = AudioCapture(ring, device)
    capture.start()
    started = time.monotonic()
    received = 0
    try:
        while time.monotonic() - started < 3.0:
            block = ring.read(timeout=0.5)
            if block is not None:
                assert block.start == received and block.lost_before == 0
                received += len(block.samples)
        elapsed = time.monotonic() - started
    finally:
        capture.stop()
    expected = elapsed * device.samplerate
    assert abs(received - expected) < 0.1 * device.samplerate  # continuidade (erro < 100 ms)
    assert capture.overflows == 0 and ring.stats().overrun_samples == 0


@pytest.mark.audio_hw
def test_loopback_speaker_to_microphone(tmp_path: Path) -> None:
    """Toca fala no alto-falante e confere que o microfone a captura (nível sobe acima do silêncio)."""
    from ddsd.audio.capture import AudioCapture, resolve_device
    from ddsd.audio.ring_buffer import AudioRingBuffer

    clip = say(tmp_path, "loop", "teste de captura do microfone, um dois três", 44_100)
    device = resolve_device("")
    ring = AudioRingBuffer(device.samplerate * 20)
    capture = AudioCapture(ring, device)
    capture.start()
    try:
        time.sleep(1.0)
        quiet = ring.read(timeout=1.0)
        subprocess.run(["afplay", str(clip)], check=True)
        loud = ring.read(timeout=1.0)
    finally:
        capture.stop()
    rms = lambda x: float(np.sqrt(np.mean(x.samples**2)))  # noqa: E731
    assert quiet is not None and loud is not None
    assert rms(loud) > 2 * rms(quiet)
