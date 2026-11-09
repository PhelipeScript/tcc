"""Teclas de controle do painel, lidas do terminal sem eco e sem Enter (modo cbreak).

| Tecla            | Ação                                                                     |
|------------------|--------------------------------------------------------------------------|
| `M` ou espaço    | pausa / retoma o microfone (pausado, nada entra e o pipeline fica ocioso) |
| `1` `2` `3`      | classificador BERTimbau / Albertina / DeBERTinha (vale da próxima fala)  |
| `C`              | próximo classificador                                                    |
| `Q`              | encerra (como o Ctrl+C)                                                  |

O modo cbreak mantém os sinais do terminal, então o Ctrl+C continua funcionando. O estado original
do terminal é restaurado ao sair, mesmo com erro. Sequências de escape (setas etc.) são ignoradas.
"""

from __future__ import annotations

import os
import select
import termios
import threading
import tty
from collections.abc import Callable
from typing import Any

from ddsd.observability.logger import get_logger
from ddsd.realtime.config import CLASSIFIER_CHOICES

log = get_logger("teclado")

#: Legenda curta exibida no painel.
LEGEND = "[M] microfone  [1] BERTimbau  [2] Albertina  [3] DeBERTinha  [C] próximo  [Q] sair"


def handle_key(pipeline: Any, key: str) -> str | None:
    """Aplica a tecla ao pipeline e devolve o nome da ação ("sair" fica com quem chamou)."""
    k = key.lower()
    if k in ("m", " "):
        pipeline.toggle_capture()
        return "microfone"
    if k in ("1", "2", "3"):
        pipeline.set_classifier(CLASSIFIER_CHOICES[int(k) - 1])
        return CLASSIFIER_CHOICES[int(k) - 1]
    if k == "c":
        # Com um TF-IDF ativo (fora do ciclo das teclas), "c" volta ao primeiro Transformer.
        current = (CLASSIFIER_CHOICES.index(pipeline.classifier_name)
                   if pipeline.classifier_name in CLASSIFIER_CHOICES else -1)
        pipeline.set_classifier(CLASSIFIER_CHOICES[(current + 1) % len(CLASSIFIER_CHOICES)])
        return "proximo_classificador"
    if k == "q":
        return "sair"
    return None


class KeyReader:
    """Thread que lê teclas de um terminal e chama `on_key(tecla)` para cada uma."""

    def __init__(self, on_key: Callable[[str], None], fd: int | None = None) -> None:
        self._on_key = on_key
        self._fd = fd
        self._saved: list[Any] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        """Liga a leitura; devolve False (e avisa) se a entrada não for um terminal."""
        if self._fd is None:
            import sys

            self._fd = sys.stdin.fileno()
        if not os.isatty(self._fd):
            log.warn("a entrada não é um terminal: as teclas de controle ficam desligadas")
            return False
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._thread = threading.Thread(target=self._run, name="teclado", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(1)
        if self._saved is not None and self._fd is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            self._saved = None

    def __enter__(self) -> KeyReader:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def _run(self) -> None:
        assert self._fd is not None
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([self._fd], [], [], 0.1)
                if not ready:
                    continue
                data = os.read(self._fd, 64).decode("utf-8", errors="ignore")
            except OSError:
                return  # terminal fechado
            for ch in keys_of(data):
                try:
                    self._on_key(ch)
                except Exception as exc:  # noqa: BLE001 - uma tecla com erro não derruba a leitura
                    log.error("tecla %r falhou: %r", ch, exc)


def keys_of(data: str) -> list[str]:
    """Teclas simples de uma leitura, sem as sequências de escape (setas: ESC [ A; F1: ESC O P…)."""
    keys, i = [], 0
    while i < len(data):
        if data[i] == "\x1b":
            i += 1
            if i < len(data) and data[i] in "[O":
                i += 1
                while i < len(data) and not "@" <= data[i] <= "~":
                    i += 1
                i += 1  # o byte final da sequência
            continue
        keys.append(data[i])
        i += 1
    return keys
