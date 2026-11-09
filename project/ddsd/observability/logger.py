"""Logger central do sistema.

`SystemLogger` expõe apenas `info`, `debug`, `warn` e `error`. Configuração
e criação ficam em funções do módulo (`configure_logging`, `get_logger`), de
modo que nenhum componente dependa de outro método além desses quatro.

Todos os loggers compartilham um único destino (`LogSink`), trocado conforme o
modo de execução:

- `ConsoleSink`: imprime cada linha no terminal (CLIs e `--no-dashboard`);
- `BufferSink`: guarda as últimas linhas num `deque` lido pelo painel Rich, sem
  imprimir nada, para não quebrar o desenho em tela cheia.

A formatação é preguiçosa, no estilo `%` do `logging` da biblioteca padrão: a
mensagem só é montada se o nível estiver habilitado. Por isso `debug()` desligado
custa apenas uma comparação, mesmo em laços quentes do áudio.
"""

from __future__ import annotations

import logging
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol

from rich.console import Console
from rich.text import Text


class LogLevel(IntEnum):
    """Níveis em ordem crescente de severidade."""

    DEBUG = 10
    INFO = 20
    WARN = 30
    ERROR = 40


@dataclass(frozen=True, slots=True)
class LogRecord:
    """Linha de log já formatada, imutável para poder circular entre threads."""

    timestamp: float
    level: LogLevel
    component: str
    message: str
    thread: str


class LogSink(Protocol):
    """Destino das linhas de log."""

    def emit(self, record: LogRecord) -> None: ...


_LEVEL_STYLE: dict[LogLevel, tuple[str, str]] = {
    LogLevel.DEBUG: ("DBG", "dim"),
    LogLevel.INFO: ("INF", "cyan"),
    LogLevel.WARN: ("AVI", "bold yellow"),
    LogLevel.ERROR: ("ERR", "bold red"),
}


def render_record(record: LogRecord) -> Text:
    """Formata uma linha de log como texto Rich (usado pelo console e pelo painel)."""
    label, style = _LEVEL_STYLE[record.level]
    clock = time.strftime("%H:%M:%S", time.localtime(record.timestamp))
    millis = int((record.timestamp % 1) * 1000)
    line = Text()
    line.append(f"{clock}.{millis:03d} ", style="dim")
    line.append(f"{label} ", style=style)
    line.append(f"[{record.component}] ", style="magenta")
    line.append(record.message, style=style if record.level >= LogLevel.WARN else "")
    return line


class ConsoleSink:
    """Imprime cada linha no terminal (stderr), protegido por lock."""

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console(stderr=True, highlight=False, soft_wrap=True)
        self._lock = threading.Lock()

    def emit(self, record: LogRecord) -> None:
        with self._lock:
            self._console.print(render_record(record))


class BufferSink:
    """Guarda as últimas `capacity` linhas para o painel, sem imprimir."""

    def __init__(self, capacity: int = 200) -> None:
        self._records: deque[LogRecord] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._counts = {level: 0 for level in LogLevel}

    def emit(self, record: LogRecord) -> None:
        with self._lock:
            self._records.append(record)
            self._counts[record.level] += 1

    def snapshot(self) -> tuple[list[LogRecord], dict[LogLevel, int]]:
        """Cópia das linhas recentes e dos totais por nível (leitura do painel)."""
        with self._lock:
            return list(self._records), dict(self._counts)


class _LoggingState:
    """Configuração global compartilhada por todos os `SystemLogger`."""

    def __init__(self) -> None:
        self.level = LogLevel.INFO
        self.sink: LogSink = ConsoleSink()


_STATE = _LoggingState()


class SystemLogger:
    """Logger de um componente. Interface pública restrita a info/debug/warn/error."""

    __slots__ = ("_component",)

    def __init__(self, component: str) -> None:
        self._component = component

    def debug(self, message: str, *args: object) -> None:
        """Detalhe para diagnóstico; desligado por padrão."""
        self._log(LogLevel.DEBUG, message, args)

    def info(self, message: str, *args: object) -> None:
        """Evento normal de operação."""
        self._log(LogLevel.INFO, message, args)

    def warn(self, message: str, *args: object) -> None:
        """Situação anormal recuperável (sobrecarga, descarte, degradação)."""
        self._log(LogLevel.WARN, message, args)

    def error(self, message: str, *args: object) -> None:
        """Falha que afeta um resultado ou exige atenção."""
        self._log(LogLevel.ERROR, message, args)

    def _log(self, level: LogLevel, message: str, args: tuple[object, ...]) -> None:
        if level < _STATE.level:
            return
        if args:
            try:
                message = message % args
            except (TypeError, ValueError):
                message = f"{message} {args!r}"
        _STATE.sink.emit(
            LogRecord(
                timestamp=time.time(),
                level=level,
                component=self._component,
                message=message,
                thread=threading.current_thread().name,
            )
        )


def get_logger(component: str) -> SystemLogger:
    """Cria o logger de um componente (todos compartilham nível e destino)."""
    return SystemLogger(component)


class _StdlibBridge(logging.Handler):
    """Encaminha `logging` e `warnings` das bibliotecas para o `SystemLogger`."""

    _MAP = (
        (logging.ERROR, LogLevel.ERROR),
        (logging.WARNING, LogLevel.WARN),
        (logging.INFO, LogLevel.INFO),
    )

    def emit(self, record: logging.LogRecord) -> None:
        level = next((ours for theirs, ours in self._MAP if record.levelno >= theirs), LogLevel.DEBUG)
        logger = SystemLogger(f"lib.{record.name.split('.')[0]}")
        try:
            text = record.getMessage()
        except Exception:  # noqa: BLE001 - mensagem malformada de terceiro não pode derrubar o sistema
            text = str(record.msg)
        # Avisos de depreciação/arquivo aberto de bibliotecas não exigem ação: ficam em debug.
        if record.name == "py.warnings" and any(k in text for k in ("FutureWarning", "DeprecationWarning",
                                                                     "ResourceWarning")):
            level = LogLevel.DEBUG
        logger._log(level, "%s", (text,))  # noqa: SLF001 - ponte interna do próprio módulo


def configure_logging(*, level: LogLevel | str = LogLevel.INFO, sink: LogSink | None = None) -> None:
    """Define nível e destino globais e redireciona o `logging` das bibliotecas.

    Args:
        level: nível mínimo (`"debug"`, `"info"`, `"warn"`, `"error"` ou `LogLevel`).
        sink: destino das linhas; `None` mantém o atual.
    """
    if isinstance(level, str):
        level = LogLevel[level.upper().replace("WARNING", "WARN")]
    _STATE.level = level
    if sink is not None:
        _STATE.sink = sink

    root = logging.getLogger()
    for handler in list(root.handlers):
        if isinstance(handler, _StdlibBridge):
            root.removeHandler(handler)
    root.addHandler(_StdlibBridge())
    # Bibliotecas ruidosas só passam de WARNING para cima, salvo em modo debug.
    root.setLevel(logging.DEBUG if level <= LogLevel.DEBUG else logging.WARNING)
    logging.captureWarnings(True)
    warnings.simplefilter("default")


def current_sink() -> LogSink:
    """Destino atualmente configurado (o painel lê o `BufferSink` por aqui)."""
    return _STATE.sink
