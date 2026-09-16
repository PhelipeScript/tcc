"""
Descrição:
    Ponto de entrada do experimento de ASR simulado (TTS → Whisper).

Uso:
    uv run python -m src.asr_eval.run_asr_eval
    uv run python -m src.asr_eval.run_asr_eval --sample-size 20  # teste de fumaça
"""

from __future__ import annotations

import argparse
import dataclasses

from src.asr_eval.config import AsrEvalConfig
from src.asr_eval.engine import run


def build_parser() -> argparse.ArgumentParser:
    """
    Descrição:
        Monta o parser de argumentos do experimento.

    Retorna:
        Parser configurado, com os padrões de `AsrEvalConfig`.
    """
    defaults = AsrEvalConfig()
    parser = argparse.ArgumentParser(
        prog="run_asr_eval",
        description="Pipeline ASR simulado (TTS -> Whisper) para os objetivos de n-best e WER.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sample-size", type=int, default=defaults.sample_size)
    parser.add_argument(
        "--n-values", type=int, nargs="+", default=list(defaults.n_values), metavar="N"
    )
    parser.add_argument("--whisper-size", default=defaults.whisper_size)
    parser.add_argument("--beam-size", type=int, default=defaults.beam_size)
    parser.add_argument("--language", default=defaults.language)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--compute-type", default=defaults.compute_type)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    return parser


def main() -> int:
    """
    Descrição:
        Roda o experimento com os parâmetros da linha de comando.

    Retorna:
        Código de saída do processo (sempre 0).
    """
    args = build_parser().parse_args()
    config = dataclasses.replace(
        AsrEvalConfig(),
        sample_size=args.sample_size,
        n_values=tuple(args.n_values),
        whisper_size=args.whisper_size,
        beam_size=args.beam_size,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        seed=args.seed,
    )
    run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
