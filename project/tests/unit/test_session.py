"""Gravação da sessão: diretório exclusivo, fechamento em qualquer saída e falhas de escrita que nunca
derrubam o pipeline."""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from ddsd.cli import build_parser
from ddsd.observability import logger
from ddsd.realtime.session import EVIDENCE_FILE, SESSION_FILE, SessionRecorder


@pytest.fixture
def restore_logging():
    """Os comandos reconfiguram o logger global; o estado anterior volta no fim do teste."""
    sink, level = logger._STATE.sink, logger._STATE.level
    yield
    logger.configure_logging(level=level, sink=sink)


def test_refuses_directory_with_previous_session(tmp_path) -> None:
    (tmp_path / EVIDENCE_FILE).write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        SessionRecorder(tmp_path, {})


def test_write_failure_is_counted_and_recording_continues(tmp_path) -> None:
    recorder = SessionRecorder(tmp_path / "s", {"x": 1})
    recorder.decision(object())  # payload inválido: a escrita falha dentro da thread
    summary = recorder.close()
    assert summary["erros"] == 1 and summary["decisoes_gravadas"] == 0 and summary["thread_encerrada"]
    session = json.loads((tmp_path / "s" / SESSION_FILE).read_text(encoding="utf-8"))
    assert session["metadados"] == {"x": 1} and session["resumo"]["erros"] == 1


def test_context_manager_writes_summary_when_the_body_fails(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="falha ao iniciar"), SessionRecorder(tmp_path / "s", {}):
        raise RuntimeError("falha ao iniciar o pipeline")
    summary = json.loads((tmp_path / "s" / SESSION_FILE).read_text(encoding="utf-8"))["resumo"]
    assert summary["thread_encerrada"] and summary["falas_gravadas"] == 0 and summary["pendentes_no_fim"] == 0


def test_close_is_idempotent(tmp_path) -> None:
    recorder = SessionRecorder(tmp_path / "s", {})
    first = recorder.close()
    written = (tmp_path / "s" / SESSION_FILE).read_bytes()
    assert recorder.close() is first
    assert (tmp_path / "s" / SESSION_FILE).read_bytes() == written


@pytest.mark.parametrize("command", ["run", "replay"])
def test_cli_rejects_used_session_dir_before_loading_models(tmp_path, capsys, restore_logging, command) -> None:
    session_dir = tmp_path / "sessao"
    session_dir.mkdir()
    (session_dir / EVIDENCE_FILE).write_text("{}\n", encoding="utf-8")
    argv = [command, "--gravar", str(session_dir)]
    if command == "replay":
        sf.write(tmp_path / "a.wav", np.zeros(16000, dtype=np.float32), 16000, subtype="PCM_16")
        argv.append(str(tmp_path / "a.wav"))
    args = build_parser().parse_args(argv)
    assert args.handler(args) == 1
    assert "já tem uma sessão gravada" in capsys.readouterr().err
    assert sorted(p.name for p in session_dir.iterdir()) == [EVIDENCE_FILE]  # nada foi sobrescrito
