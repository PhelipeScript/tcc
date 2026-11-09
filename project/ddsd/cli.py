"""Ponto de entrada `ddsd` (`uv run ddsd <comando> ...`).

Cada pacote expõe uma função `register(subparsers)` em `<pacote>/commands.py`.
Os handlers importam suas dependências pesadas (torch, mlx, sounddevice)
somente quando são executados, para que `ddsd --help` e os comandos leves
abram rápido.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence

#: Módulos que registram subcomandos, na ordem em que aparecem na ajuda.
_COMMAND_MODULES: tuple[str, ...] = (
    "ddsd.corpus.commands",
    "ddsd.training.commands",
    "ddsd.realtime.commands",
    "ddsd.evaluation.commands",
)


def build_parser() -> argparse.ArgumentParser:
    """Monta o parser com os subcomandos de todos os pacotes."""
    parser = argparse.ArgumentParser(
        prog="ddsd",
        description="DDSD textual em PT-BR: corpus, treino, avaliação e pipeline em tempo real.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<comando>")
    for module_name in _COMMAND_MODULES:
        module = importlib.import_module(module_name)
        module.register(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Executa o subcomando escolhido e devolve o código de saída do processo."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 2
    return int(handler(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
