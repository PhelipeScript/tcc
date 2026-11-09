"""Logger (API restrita, nível, formatação preguiçosa) e painel (quadro em tela cheia a partir do snapshot)."""

from __future__ import annotations

import pytest
from rich.console import Console

from rich.cells import cell_len

from ddsd.observability.dashboard import MIN_HEIGHT, MIN_WIDTH, Dashboard, build_frame
from ddsd.observability.logger import BufferSink, LogLevel, LogRecord, SystemLogger, configure_logging, get_logger
from ddsd.observability.status import StatusBoard


@pytest.fixture
def sink():
    sink = BufferSink(50)
    configure_logging(level="info", sink=sink)
    yield sink
    configure_logging(level="info")


def test_logger_public_api_is_only_four_methods() -> None:
    public = {name for name in dir(SystemLogger) if not name.startswith("_")}
    assert public == {"info", "debug", "warn", "error"}


def test_level_filtering_and_formatting(sink: BufferSink) -> None:
    log = get_logger("teste")
    log.debug("invisível %s", 1)
    log.info("fala %s em %.1f s", "U0001", 2.345)
    log.warn("aviso")
    log.error("erro %d", 7)
    records, counts = sink.snapshot()
    assert [r.message for r in records] == ["fala U0001 em 2.3 s", "aviso", "erro 7"]
    assert counts[LogLevel.DEBUG] == 0 and counts[LogLevel.ERROR] == 1
    assert all(r.component == "teste" for r in records)


def test_debug_disabled_does_not_format(sink: BufferSink) -> None:
    class Explodes:
        def __str__(self) -> str:  # se fosse formatado, o teste falharia
            raise AssertionError("formatado com debug desligado")

    get_logger("teste").debug("valor %s", Explodes())
    assert sink.snapshot()[0] == []


def test_bad_format_args_never_raise(sink: BufferSink) -> None:
    get_logger("teste").info("faltam %s %s", "um")
    assert "faltam" in sink.snapshot()[0][0].message


LONG_TEXT = ("abre o navegador e procura aquela receita de bolo de chocolate com cobertura de brigadeiro que "
             "a minha tia mandou ontem no grupo da família, por favor, e depois salva nos favoritos")


def _board(state: str) -> StatusBoard:
    """Estados do painel montados pela mesma API que o pipeline usa."""
    status = StatusBoard()
    status.set("pipeline", estado="ouvindo", atividade="ocioso", classificador="bertimbau", cpu_pct=4,
               execucao="bertimbau-s44", limiar=0.685, backend="mlx", sessao="20261103-103201",
               modelo_whisper="mlx-community/whisper-large-v3-turbo")
    status.set("captura", estado="ativa", dispositivo="Microfone do MacBook Air com um nome bem comprido demais",
               taxa=48000, nivel_dbfs=-18.4, pico_dbfs=-8.2, nivel_hist=tuple(range(-60, 0, 3)), overflows=0)
    status.set("vad", estado="silêncio")
    status.set("frontend", atraso_ms=6.0)
    status.set("memoria", rss_gb=1.24)
    status.set("workers", asr="ocioso", fuzzy="ocioso", **{"clf-bertimbau": "ocioso"})
    status.observe("whisper.final", 0.42)
    status.observe("fuzzy", 0.0012)
    status.observe("clf.bertimbau", 0.032)
    if state == "ocioso":
        return status
    # Fala anterior já decidida (vai para "falas anteriores" quando a próxima abre).
    status.utterance_opened("U0006", 10.0, 0.0)
    status.utterance_update("U0006", state="decidida", label="NDD", p_dd=0.08, threshold=0.685,
                            reliability_score=0.52, reliability_label="média", final_text=" Nossa, travou de novo.",
                            duration_s=2.5, latency_s=1.1, reason="classificador")
    status.utterance_opened("U0007", 20.0, 0.0)
    status.set("whisper", ultima_tarefa="U0006-F")
    status.set("fuzzy", ultima_tarefa="U0006-F")
    status.set("classificador", ultima_tarefa="U0006-F")
    if state == "fala":
        status.set("pipeline", atividade="fala")
        status.set("vad", estado="fala", fala="U0007", duracao_s=10.8)
        status.set("workers", asr="ocioso", fuzzy="ocioso", **{"clf-bertimbau": "ocioso"})
        return status
    status.utterance_update("U0007", state="transcrevendo", duration_s=10.8)
    if state == "processando":
        status.set("pipeline", atividade="processando")
        status.set("workers", asr="U0007-F", fuzzy="ocioso", **{"clf-bertimbau": "ocioso"})
        return status
    status.observe("decisao", 1.21)
    status.utterance_update("U0007", state="decidida", label="DD", reason="classificador", p_dd=0.97, threshold=0.685,
                            reliability_score=0.31, reliability_label="baixa", reliability_alert=True,
                            final_text=" " + LONG_TEXT, final_normalized=LONG_TEXT, latency_s=1.21)
    if state == "sobreposta":  # U0007 ainda processando quando U0008 abre
        status.utterance_update("U0007", state="analisando")
        status.utterance_opened("U0008", 32.0, 0.0)
        status.set("pipeline", atividade="fala")
        status.set("vad", estado="fala", fala="U0008", duracao_s=2.8)
    return status


def _records(n: int = 6) -> list[LogRecord]:
    return [LogRecord(float(i), LogLevel.WARN if i == n - 1 else LogLevel.INFO, "pipeline",
                      f"evento {i} com uma mensagem comprida o bastante para ser cortada no painel " * 2, "t")
            for i in range(n)]


def _plain(state: str, width: int, height: int) -> list[str]:
    lines = build_frame(_board(state).snapshot(), _records(), width, height, now_wall=0.0)
    assert len(lines) <= height, (state, width, height, len(lines))
    for line in lines:
        assert cell_len(line.plain) == width, (state, width, height, line.plain)
    return [line.plain for line in lines]


@pytest.mark.parametrize("width", [100, 120, 160])
@pytest.mark.parametrize("height", [34, 40, 50])
@pytest.mark.parametrize("state", ["ocioso", "fala", "processando", "decidida", "sobreposta"])
def test_frame_fits_the_terminal_exactly(state: str, width: int, height: int) -> None:
    text = "\n".join(_plain(state, width, height))
    for expected in ("DDSD  /  DEVICE-DIRECTED SPEECH DETECTION", "ENTRADA DE ÁUDIO", "ESTADO DO PIPELINE",
                     "FALA COMPLETA"):
        assert expected in text, (state, width, height, expected)
    assert text.splitlines()[0].startswith("┌") and text.splitlines()[-1].startswith("└")


def test_idle_frame_shows_that_whisper_and_classifier_are_stopped() -> None:
    text = "\n".join(_plain("ocioso", 120, 50))
    for expected in ("○ OCIOSO • Whisper e BERTimbau parados", "SESSÃO 20261103-103201", "CLASSIFICADOR: BERTIMBAU",
                     "WHISPER: LARGE-V3-TURBO", "mono • 48000 Hz • float32", "-18,4 dBFS", "-8,2 dBFS", "○ SILÊNCIO",
                     "AGUARDANDO A PRIMEIRA FALA"):
        assert expected in text, expected


def test_speech_frame_shows_capture_progress_and_pending_decision() -> None:
    text = "\n".join(_plain("fala", 120, 50))
    for expected in ("● EXECUTANDO", "● FALA DETECTADA", "Fala atual  : 00:10,8", "CAPTURA EM ANDAMENTO",
                     "aguardando o fim da fala", "PENDENTE - será executada sobre a transcrição da fala completa",
                     "PENDENTE - a decisão sai da fala completa", "U0006 concluída", "U0006",
                     "Whisper (fala)", "420 ms"):
        assert expected in text, expected


def test_processing_and_decided_frames() -> None:
    processing = "\n".join(_plain("processando", 120, 50))
    assert "● TRANSCREVENDO U0007" in processing and "TRANSCREVENDO A FALA COMPLETA" in processing
    decided = "\n".join(_plain("decidida", 160, 50))
    for expected in ("DECIDIDA", "● DD - CONSOLIDADA", "DD • p 0,97 (limiar 0,685)", "0,31 baixa",
                     "Entrada do classificador", "minha tia mandou",
                     "1,21 s após o fim da fala", "ALERTA: transcrição pouco confiável", "FALAS ANTERIORES",
                     "U0006", "NDD • p 0,08", "EVENTOS RECENTES", "AVISO"):
        assert expected in decided, expected
    assert "…" in decided  # a transcrição longa é cortada com reticências, sem quebrar o quadro


def test_overlapping_utterances_follow_the_new_one() -> None:
    text = "\n".join(_plain("sobreposta", 160, 50))
    assert "FALA COMPLETA  •  U0008  •  CAPTURA EM ANDAMENTO" in text and "U0007" in text
    assert "processando…" in text  # a fala anterior ainda sem decisão aparece em "falas anteriores"


def test_short_terminal_trims_events_first_and_never_scrolls() -> None:
    tall = "\n".join(_plain("decidida", 120, 50))
    short = "\n".join(_plain("decidida", 120, 34))
    assert tall.count("evento") > short.count("evento")
    assert "ESTADO DO PIPELINE" in short and "● DD - CONSOLIDADA" in short


def test_too_small_terminal_shows_a_notice() -> None:
    lines = build_frame(_board("decidida").snapshot(), [], MIN_WIDTH - 10, MIN_HEIGHT, now_wall=0.0)
    assert "Terminal pequeno" in lines[0].plain and all(cell_len(x.plain) == MIN_WIDTH - 10 for x in lines)


def test_dashboard_renderable_uses_the_console_size(sink: BufferSink) -> None:
    get_logger("teste").warn("fila cheia")
    console = Console(record=True, width=130, height=40, force_terminal=True, color_system=None)
    console.print(Dashboard(_board("fala"), sink).render())
    text = console.export_text()  # (export_text esvazia o buffer gravado)
    lines = text.rstrip("\n").splitlines()
    assert len(lines) <= 40 and all(cell_len(line) == 130 for line in lines)
    assert "AVISO  fila cheia" in text  # o log real entra em "eventos recentes"


@pytest.mark.parametrize("width", [100, 120, 160])
def test_frame_shows_the_key_legend_and_the_control_states(width: int) -> None:
    status = _board("ocioso")
    status.set("pipeline", atividade="pausado", carregando="albertina")
    status.set("captura", estado="pausada")
    lines = build_frame(status.snapshot(), [], width, 40, now_wall=0.0)
    assert all(cell_len(line.plain) == width for line in lines) and len(lines) <= 40
    top = lines[0].plain
    assert top.startswith("┌") and "[M] microfone" in top and "[1] BERTimbau" in top and "[Q] sair" in top
    text = "\n".join(line.plain for line in lines)
    for expected in ("○ PAUSADO • microfone fechado, sistema ocioso", "○ PAUSADA (tecla M retoma)",
                     "● carregando Albertina…"):
        assert expected in text, (width, expected)
