"""
Descrição:
    Ponto de entrada da validação externa (CORAA, NURC-SP, TAGARELA).

Uso:
    uv run python -m src.external_eval.run_external_eval
    uv run python -m src.external_eval.run_external_eval --corpora coraa --sample-size 100
"""

from __future__ import annotations

import argparse
import dataclasses

from src.external_eval.config import CORPUS_NAMES, ExternalEvalConfig
from src.external_eval.engine import run


def build_parser() -> argparse.ArgumentParser:
    """
    Descrição:
        Monta o parser de argumentos da validação externa.

    Retorna:
        Parser configurado, com os padrões de `ExternalEvalConfig`.
    """
    defaults = ExternalEvalConfig()
    parser = argparse.ArgumentParser(
        prog="run_external_eval",
        description="Valida os classificadores treinados contra fala real (CORAA/NURC-SP/TAGARELA).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--corpora", nargs="+", choices=CORPUS_NAMES, default=list(defaults.corpora)
    )
    parser.add_argument("--sample-size", type=int, default=defaults.sample_size)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    return parser


def main() -> int:
    """
    Descrição:
        Roda a validação externa com os parâmetros da linha de comando.

    Retorna:
        Código de saída do processo (sempre 0).
    """
    args = build_parser().parse_args()
    config = dataclasses.replace(
        ExternalEvalConfig(),
        corpora=tuple(args.corpora),
        sample_size=args.sample_size,
        seed=args.seed,
    )
    run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
