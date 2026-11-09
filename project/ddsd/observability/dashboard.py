"""Painel do pipeline em tempo real: tela cheia (Rich `Live` com `screen=True`).

Segue o esboço do autor: um quadro único, com seções separadas por linhas e títulos em maiúsculas.
- Largura: a partir de `WIDE` colunas, o alto do quadro tem 3 colunas (entrada, estado,
  desempenho); abaixo disso, 2 colunas, como no esboço.
- Altura: o quadro nunca passa da altura do terminal. Se faltar espaço, corta, nesta ordem,
  eventos, falas anteriores, desempenho e entrada. Estado do pipeline e fala completa ficam sempre.
- `build_frame()` é uma função pura do snapshot do `StatusBoard` e das últimas linhas do log, então o
  painel nunca bloqueia captura nem inferência e é testável sem terminal.

A borda de cima mostra as teclas de controle (`realtime/keys.py`): microfone e classificador.

Cores do diagrama do pipeline: captura em azul, VAD em amarelo, Whisper em verde, fala completa em
laranja e decisão em roxo. Só usa caracteres de largura 1 (sem emoji), para as bordas ficarem alinhadas.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from rich.cells import cell_len
from rich.console import Console, ConsoleOptions, RenderResult
from rich.text import Text

from ddsd.observability.logger import BufferSink, LogLevel, LogRecord
from ddsd.observability.status import StatusBoard, StatusSnapshot, UtteranceView

BLUE, YELLOW, GREEN, ORANGE, PURPLE, RED = "#3985D6", "#D6A900", "#72B52D", "#F07800", "#A276C8", "#E5484D"
BORDER = "grey50"

#: Largura a partir da qual o alto do quadro usa 3 colunas.
WIDE = 140
#: Abaixo disso, o painel mostra só um aviso com o estado e a última decisão.
MIN_WIDTH, MIN_HEIGHT = 96, 26

CLASSIFIER_LABELS = {"bertimbau": "BERTimbau", "albertina": "Albertina", "debertinha": "DeBERTinha",
                     "baseline_logreg": "TF-IDF + Regressão Logística", "baseline_svm": "TF-IDF + SVM"}
REASON_TEXT = {"sem_texto": "sem texto", "falha_asr": "falha no ASR",
               "falha_classificador": "classificador sem resposta", "classificador": ""}
LEVEL_LABEL = {LogLevel.DEBUG: ("DEBUG", "dim"), LogLevel.INFO: ("INFO ", "dim"),
               LogLevel.WARN: ("AVISO", f"bold {YELLOW}"), LogLevel.ERROR: ("ERRO ", f"bold {RED}")}
SPARK = "▁▂▃▄▅▆▇█"


def classifier_label(name: str | None) -> str:
    return CLASSIFIER_LABELS.get(name or "", name or "-")


# ------------------------------------------------------------------ formatos
def _num(value: float | None, digits: int = 2) -> str:
    """Número com vírgula decimal (PT-BR); "-" quando ausente."""
    return "-" if value is None else f"{value:.{digits}f}".replace(".", ",")


def _clock(seconds: float | None, decimals: int = 1) -> str:
    """Duração como mm:ss,d."""
    if seconds is None:
        return "-"
    seconds = max(seconds, 0.0)
    minutes, rest = divmod(seconds, 60)
    width = 3 + decimals if decimals else 2
    return f"{int(minutes):02d}:{rest:0{width}.{decimals}f}".replace(".", ",")


def _ms(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    ms = seconds * 1000
    return f"{_num(ms, 1)} ms" if ms < 10 else f"{ms:.0f} ms"


def _one_line(text: str | None) -> str:
    return " ".join((text or "").split())


def _quote(text: str | None) -> str:
    text = _one_line(text)
    return f'"{text}"' if text else "-"


def _kv(label: str, value: Text | str, width: int, style: str = "") -> Text:
    line = Text(f"{label:<{width}}", style="dim")
    line.append(": ", style="dim")
    line.append_text(value if isinstance(value, Text) else Text(str(value), style=style))
    return line


def _dot(text: str, style: str, filled: bool = True) -> Text:
    return Text(f"{'●' if filled else '○'} {text}", style=style)


def _decision(label: str | None, p_dd: float | None) -> Text:
    if label is None:
        return Text("-", style="dim")
    text = Text(label, style=f"bold {PURPLE}" if label == "DD" else "bold")
    if p_dd is not None:
        text.append(f" • p {_num(p_dd)}", style="dim")
    return text


def _fuzzy(score: float | None, label: str | None) -> Text:
    if score is None:
        return Text("-", style="dim")
    style = {"alta": GREEN, "média": YELLOW, "baixa": f"bold {RED}"}.get(label or "", "")
    return Text(f"{_num(score)} {label or ''}".rstrip(), style=style)


def _spark(history: tuple[float, ...] | list[float] | None, width: int) -> Text:
    if not history:
        return Text("-", style="dim")
    bars = "".join(SPARK[min(len(SPARK) - 1, max(0, int((db + 60) / 60 * len(SPARK))))] for db in history[-width:])
    return Text(bars, style=BLUE)


# -------------------------------------------------------------- quadro
@dataclass
class Section:
    title: Text
    lines: list[Text]


def _cell(text: Text, width: int) -> Text:
    """Exatamente `width` células: corta com reticências e completa com espaços."""
    cell = text.copy()
    cell.no_wrap = True
    cell.truncate(width, overflow="ellipsis", pad=True)
    return cell


def _widths(total: int, columns: int) -> list[int]:
    """Larguras externas das colunas (com 1 espaço de cada lado) para uma linha de `total` células."""
    free = total - (columns + 1)
    base, extra = divmod(free, columns)
    return [base + (1 if i < extra else 0) for i in range(columns)]


def _boundaries(widths: list[int]) -> set[int]:
    positions, x = set(), 0
    for w in widths[:-1]:
        x += w + 1
        positions.add(x)
    return positions


def _separator(width: int, above: set[int] | None, below: set[int] | None) -> Text:
    left, right, fill = ("┌", "┐", "─") if above is None else (("└", "┘", "─") if below is None else ("├", "┤", "─"))
    chars = []
    for x in range(1, width - 1):
        up, down = above is not None and x in above, below is not None and x in below
        chars.append("┼" if up and down else "┴" if up else "┬" if down else fill)
    return Text(left + "".join(chars) + right, style=BORDER)


def _row(sections: list[Section], width: int) -> tuple[list[Text], set[int]]:
    widths = _widths(width, len(sections))
    bodies = [[Text.assemble(s.title), *s.lines] for s in sections]
    height = max(len(b) for b in bodies)
    lines = []
    for i in range(height):
        line = Text("│", style=BORDER)
        for body, w in zip(bodies, widths, strict=True):
            content = body[i] if i < len(body) else Text("")
            line.append(" ")
            line.append_text(_cell(content, w - 2))
            line.append(" │", style=BORDER)
        lines.append(line)
    return lines, _boundaries(widths)


KEYS_LEGEND = " [M] microfone  [1] BERTimbau  [2] Albertina  [3] DeBERTinha  [C] próximo  [Q] sair "


def compose(rows: list[list[Section]], width: int, hint: str = " Ctrl+C encerra ",
            legend: str = KEYS_LEGEND) -> list[Text]:
    """Quadro único: as seções de cada linha lado a lado, com junções ┬ ┴ ┼ nas divisões.

    `legend` (as teclas) vai na borda de cima; `hint`, na de baixo, à direita, quando não cruza junção.
    """
    out: list[Text] = []
    previous: set[int] | None = None
    for i, sections in enumerate(rows):
        lines, bounds = _row(sections, width)
        separator = _separator(width, None if i == 0 else previous, bounds)
        if i == 0 and legend and cell_len(legend) <= width - 4 and not bounds:
            plain = separator.plain
            separator = Text.assemble((plain[:2], BORDER), (legend, "bold"), (plain[2 + cell_len(legend):], BORDER))
        out.append(separator)
        out.extend(lines)
        previous = bounds
    bottom = _separator(width, previous, None)
    start = width - 2 - cell_len(hint)
    if start > 1 and set(bottom.plain[start:width - 2]) == {"─"}:
        bottom = Text.assemble((bottom.plain[:start], BORDER), (hint, "dim"), (bottom.plain[width - 2:], BORDER))
    out.append(bottom)
    return out


def _title(*parts: str, style: str = "bold") -> Text:
    return Text("  •  ".join(parts), style=style)


# ------------------------------------------------------------ seções
def _activity(s: StatusSnapshot) -> str:
    return s.components.get("pipeline", {}).get("atividade", "ocioso")


def _header(s: StatusSnapshot, now_wall: float, width: int) -> Section:
    p = s.components.get("pipeline", {})
    clf = classifier_label(p.get("classificador"))
    state = p.get("estado", "-")
    if state != "ouvindo":
        badge = _dot(str(state).upper(), f"bold {YELLOW}")
    elif _activity(s) == "pausado":
        badge = _dot("PAUSADO • microfone fechado, sistema ocioso", f"bold {YELLOW}", filled=False)
    elif _activity(s) == "ocioso":
        badge = _dot(f"OCIOSO • Whisper e {clf} parados", f"bold {BLUE}", filled=False)
    else:
        badge = _dot("EXECUTANDO", f"bold {GREEN}")
    right = Text(time.strftime("%H:%M:%S", time.localtime(now_wall)) + "  ", style="dim")
    right.append_text(badge)
    left = Text("DDSD  /  DEVICE-DIRECTED SPEECH DETECTION", style="bold")
    if cell_len(left.plain) + 2 + cell_len(right.plain) > width - 4:  # o estado importa mais que o título
        left = Text("DDSD", style="bold")
    gap = max(1, width - 4 - cell_len(left.plain) - cell_len(right.plain))
    first = Text.assemble(left, " " * gap, right)
    model = str(p.get("modelo_whisper", "-")).rsplit("/", 1)[-1].removeprefix("whisper-").removeprefix(
        "faster-whisper-").upper()
    seed = str(p.get("execucao", "-")).rsplit("-", 1)[-1]
    second = Text(f"SESSÃO {p.get('sessao', '-')}  •  CLASSIFICADOR: {clf.upper()} ({seed}, limiar "
                  f"{_num(p.get('limiar'), 3)})  •  WHISPER: {model} ({p.get('backend', '-')})", style="dim")
    return Section(first, [second])


def _entrada(s: StatusSnapshot, width: int) -> Section:
    c, vad = s.components.get("captura", {}), s.components.get("vad", {})
    rate = c.get("taxa")
    speaking = vad.get("estado") == "fala"
    lines = [
        _kv("Dispositivo", str(c.get("dispositivo", "-")), 12),
        _kv("Formato", f"mono • {rate} Hz • float32" if rate else "-", 12),
        _kv("Nível RMS", f"{_num(c.get('nivel_dbfs'), 1)} dBFS", 12),
        _kv("Pico (1 s)", f"{_num(c.get('pico_dbfs'), 1)} dBFS", 12),
        _kv("Entrada", _spark(c.get("nivel_hist"), max(4, width - 20)), 12),
        _kv("Fala atual", Text(_clock(vad.get("duracao_s")) if speaking else "-",
                               style=f"bold {ORANGE}" if speaking else "dim"), 12),
    ]
    return Section(_title("ENTRADA DE ÁUDIO", style=f"bold {BLUE}"), lines)


def _component(s: StatusSnapshot, worker: str, last_key: str, verb: str) -> Text:
    busy = s.components.get("workers", {}).get(worker)
    last = s.components.get(last_key, {}).get("ultima_tarefa")
    short = lambda task: str(task).removesuffix("-F")  # noqa: E731 - a fala (U0007)
    if busy and busy != "ocioso":
        return _dot(f"{verb} {short(busy)}", f"bold {GREEN}")
    if _activity(s) in ("ocioso", "pausado"):
        text = _dot("OCIOSO", "dim", filled=False)
        if last:
            text.append(f" (última: {short(last)})", style="dim")
        return text
    return _dot(f"{short(last)} concluída" if last else "aguardando", GREEN if last else "dim")


def _estado(s: StatusSnapshot) -> Section:
    p, c, vad = s.components.get("pipeline", {}), s.components.get("captura", {}), s.components.get("vad", {})
    capture = c.get("estado", "-")
    vad_state = vad.get("estado", "-")
    vad_text = {"fala": _dot("FALA DETECTADA", f"bold {ORANGE}"),
                "candidato": _dot("POSSÍVEL FALA", YELLOW),
                "silêncio": _dot("SILÊNCIO", "dim", filled=False)}.get(vad_state, _dot(str(vad_state), "dim"))
    clf_worker = f"clf-{p.get('classificador', 'bertimbau')}"
    if capture == "ativa":
        capture_text = _dot("ATIVA", f"bold {GREEN}")
    elif capture == "pausada":
        capture_text = _dot("PAUSADA (tecla M retoma)", f"bold {YELLOW}", filled=False)
    else:
        capture_text = _dot(str(capture).upper(), f"bold {RED}")
    classifier = _component(s, clf_worker, "classificador", "CLASSIFICANDO")
    if p.get("carregando"):
        classifier = _dot(f"carregando {classifier_label(p.get('carregando'))}…", f"bold {YELLOW}")
    lines = [
        _kv("Captura", capture_text, 12),
        _kv("Silero VAD", vad_text, 12),
        _kv("Whisper", _component(s, "asr", "whisper", "TRANSCREVENDO"), 12),
        _kv("Fuzzy", _component(s, "fuzzy", "fuzzy", "ANALISANDO"), 12),
        _kv(classifier_label(p.get("classificador")), classifier, 12),
    ]
    return Section(_title("ESTADO DO PIPELINE", style=f"bold {YELLOW}"), lines)


def _desempenho(s: StatusSnapshot) -> Section:
    p, f, m = s.components.get("pipeline", {}), s.components.get("frontend", {}), s.components.get("memoria", {})
    lat = s.latencies
    last = lambda name: lat[name][3] if name in lat else None  # noqa: E731
    clf = classifier_label(p.get("classificador"))
    queued = sum(s.components.get(f"fila.{n}", {}).get("pendentes", 0)
                 for n in ("asr", "fuzzy", p.get("classificador", "bertimbau")))
    decision = Text(_num(last("decisao")) + " s" if "decisao" in lat else "-")
    if "decisao" in lat:
        decision.append(f" (p95 {_num(lat['decisao'][2])} s)", style="dim")
    lines = [
        _kv("Captura → VAD", f"{_num(f.get('atraso_ms'), 1)} ms" if f.get("atraso_ms") is not None else "-", 18),
        _kv("Whisper (fala)", _ms(last("whisper.final")), 18),
        _kv(f"Fuzzy / {clf}", f"{_ms(last('fuzzy'))} / {_ms(last(f'clf.{p.get('classificador')}'))}", 18),
        _kv("Fala → decisão", decision, 18),
        _kv("Processo", f"CPU {p.get('cpu_pct', '-')}% • {_num(m.get('rss_gb'), 1)} GB • fila {queued}", 18),
    ]
    return Section(_title("DESEMPENHO EM TEMPO REAL", style=f"bold {BLUE}"), lines)


def _fala_completa(s: StatusSnapshot, u: UtteranceView | None) -> Section:
    vad = s.components.get("vad", {})
    if u is None:
        return Section(_title("FALA COMPLETA", "AGUARDANDO A PRIMEIRA FALA", style=f"bold {ORANGE}"),
                       [Text("Fale normalmente: o pipeline detecta a fala sozinho (sem palavra de ativação).",
                             style="dim")])
    state = u.state
    title_state = {"capturando": "CAPTURA EM ANDAMENTO", "transcrevendo": "TRANSCREVENDO A FALA COMPLETA",
                   "analisando": "CLASSIFICANDO", "decidida": "DECIDIDA"}.get(state, state.upper())
    title = _title("FALA COMPLETA", u.utterance, title_state, style=f"bold {ORANGE}")
    capturing = state == "capturando"
    audio = Text(_clock(vad.get("duracao_s") if capturing else u.duration_s))
    audio.append(" • VAD: ", style="dim")
    audio.append("fala contínua" if capturing else "fim da fala")
    audio.append(" • Whisper: ", style="dim")
    audio.append_text({"capturando": Text("aguardando o fim da fala", style="dim"),
                       "transcrevendo": Text("transcrevendo…", style=GREEN)}.get(state, Text("concluído", style=GREEN)))
    if state == "decidida":
        reason = REASON_TEXT.get(u.reason or "", u.reason or "")
        classification = _decision(u.label, u.p_dd)
        if u.threshold is not None:
            classification.append(f" (limiar {_num(u.threshold, 3)})", style="dim")
        if reason:
            classification.append(f"  •  {reason}", style="bold")
        fuzzy = _fuzzy(u.reliability_score, u.reliability_label)
        if u.reliability_alert:
            fuzzy.append("  ALERTA: transcrição pouco confiável", style=f"bold {RED}")
        decision = Text(f"● {u.label} - CONSOLIDADA", style=f"bold {PURPLE}" if u.label == "DD" else "bold")
        if u.latency_s is not None:
            decision.append(f" • {_num(u.latency_s)} s após o fim da fala", style="dim")
    else:
        classification = Text("PENDENTE - será executada sobre a transcrição da fala completa", style=YELLOW)
        fuzzy = Text("-", style="dim")
        decision = Text("PENDENTE - a decisão sai da fala completa", style=YELLOW)
    lines = [_kv("Áudio acumulado", audio, 24),
             _kv("Transcrição (RAW)", Text(_quote(u.final_text), style=f"bold {ORANGE}") if u.final_text
                 else Text("-", style="dim"), 24),
             _kv("Entrada do classificador", _quote(u.final_normalized) if u.final_normalized
                 else Text("-", style="dim"), 24),
             _kv("Classificação", classification, 24), _kv("Fuzzy", fuzzy, 24),
             _kv("Decisão", decision, 24)]
    return Section(title, lines)


def _anteriores(utterances: list[UtteranceView], max_rows: int) -> Section:
    lines = []
    for u in reversed(utterances[-1 - max_rows:-1]):
        line = Text(f"{u.utterance} ", style="bold")
        line.append(time.strftime("%H:%M:%S", time.localtime(u.opened_wall)) + "  ", style="dim")
        if u.state == "decidida":
            line.append_text(_decision(u.label, u.p_dd))
            line.append("  fuzzy ", style="dim")
            line.append_text(_fuzzy(u.reliability_score, u.reliability_label))
        else:
            line.append("processando…", style=GREEN)
        line.append("  " + _quote(u.final_text), style="dim")
        lines.append(line)
    return Section(_title("FALAS ANTERIORES", style="bold"), lines or [Text("-", style="dim")])


def _eventos(records: list[LogRecord], max_rows: int) -> Section:
    lines = []
    for record in records[-max_rows:]:
        label, style = LEVEL_LABEL.get(record.level, ("INFO ", "dim"))
        line = Text(time.strftime("%H:%M:%S", time.localtime(record.timestamp)) + "  ", style="dim")
        line.append(label + "  ", style=style)
        line.append(_one_line(record.message), style=style if record.level >= LogLevel.WARN else "")
        lines.append(line)
    title = _title("EVENTOS RECENTES", f"ÚLTIMAS {max_rows} LINHAS", style="bold")
    return Section(title, lines or [Text("-", style="dim")])


def _too_small(s: StatusSnapshot, width: int, height: int) -> list[Text]:
    u = s.utterances[-1] if s.utterances else None
    lines = [Text(f"Terminal pequeno para o painel: {width}×{height} (mínimo {MIN_WIDTH}×{MIN_HEIGHT}).",
                  style=f"bold {YELLOW}"),
             Text(KEYS_LEGEND.strip(), style="dim"),
             Text(f"Estado: {_activity(s).upper()}  •  {classifier_label(s.components.get('pipeline', {}).get('classificador'))}")]
    if u is not None and u.state == "decidida":
        lines.append(Text(f"Última decisão: {u.utterance} {u.label} {_quote(u.final_text)}"))
    return [_cell(line, width) for line in lines][:height]


def _layouts(s: StatusSnapshot, records: list[LogRecord], width: int, now_wall: float):
    """Layouts candidatos, do mais completo ao mais enxuto; o primeiro que couber na altura vence.

    A ordem do corte protege o essencial: primeiro saem eventos e falas anteriores, depois o desempenho
    e a entrada de áudio. Estado do pipeline e fala completa ficam sempre.
    """
    u = s.utterances[-1] if s.utterances else None
    header = _header(s, now_wall, width)
    two, three = _widths(width, 2)[0] - 2, _widths(width, 3)[0] - 2
    fala = [_fala_completa(s, u)]
    if width >= WIDE:
        top = [_entrada(s, three), _estado(s), _desempenho(s)]
        for history, events in ((7, 7), (6, 6), (5, 5), (4, 4), (3, 3), (2, 2), (1, 1)):
            yield [[header], top, fala, [_anteriores(s.utterances, history), _eventos(records, events)]]
        yield [[header], top, fala]
    else:
        top = [_entrada(s, two), _estado(s)]
        for history, events in ((6, 6), (5, 5), (4, 4), (3, 3), (2, 2), (1, 1)):
            yield [[header], top, fala, [_desempenho(s), _anteriores(s.utterances, history)],
                   [_eventos(records, events)]]
        for history in (5, 4, 3, 2, 1):
            yield [[header], top, fala, [_desempenho(s), _anteriores(s.utterances, history)]]
        yield [[header], top, fala, [_desempenho(s)]]
        yield [[header], top, fala]
    yield [[header], [_estado(s)], fala]


def build_frame(s: StatusSnapshot, records: list[LogRecord], width: int, height: int,
                now_wall: float | None = None) -> list[Text]:
    """Monta o quadro com no máximo `height` linhas de exatamente `width` células (nunca rola)."""
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        return _too_small(s, width, height)
    now_wall = time.time() if now_wall is None else now_wall
    frame: list[Text] = []
    for rows in _layouts(s, records, width, now_wall):
        frame = compose(rows, width)
        if len(frame) <= height:
            return frame
    return frame[: height - 1] + [frame[-1]]


class Dashboard:
    """Renderable do `rich.live.Live`: refaz o quadro no tamanho atual do terminal a cada atualização."""

    def __init__(self, status: StatusBoard, sink: BufferSink | None) -> None:
        self.status = status
        self.sink = sink

    def render(self) -> Dashboard:
        return self

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        records = self.sink.snapshot()[0] if self.sink is not None else []
        height = options.height or console.size.height
        yield from build_frame(self.status.snapshot(), records, options.max_width, height)
