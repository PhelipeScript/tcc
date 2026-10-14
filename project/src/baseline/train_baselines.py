"""
Descrição:
    Ponto de entrada dos baselines clássicos: treina SVM e Regressão
    Logística sobre TF-IDF e imprime a tabela comparativa entre os dois.

Uso:
    uv run python -m src.baseline.train_baselines
    uv run python -m src.baseline.train_baselines --only svm
    uv run python -m src.baseline.train_baselines --max-train-samples 2000  # smoke test
"""

from __future__ import annotations

import argparse
import dataclasses

from src.baseline.config import MODEL_KEYS, BaselineConfig
from src.baseline.engine import run
from src.data_prep import console
from src.training import reporting


def build_parser() -> argparse.ArgumentParser:
    """
    Descrição:
        Monta o parser de argumentos dos baselines.

    Retorna:
        Parser configurado, com os padrões de `BaselineConfig`.
    """
    defaults = BaselineConfig()
    parser = argparse.ArgumentParser(
        prog="train_baselines",
        description="Treina os baselines TF-IDF + SVM/Regressão Logística.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--only", choices=MODEL_KEYS, default=None, help="Roda um único baseline."
    )
    parser.add_argument("--cv", type=int, default=defaults.cv, help="Folds da busca de C.")
    parser.add_argument(
        "--max-features", type=int, default=defaults.max_features, help="Teto do vocabulário TF-IDF."
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Limita o treino a N exemplos (teste de fumaça).",
    )
    return parser


def main() -> int:
    """
    Descrição:
        Roda os baselines pedidos na linha de comando e imprime a tabela
        comparativa entre eles.

    Retorna:
        Código de saída do processo (sempre 0; falhas propagam como exceção).
    """
    args = build_parser().parse_args()
    config = dataclasses.replace(
        BaselineConfig(),
        cv=args.cv,
        max_features=args.max_features,
        max_train_samples=args.max_train_samples,
    )

    model_keys = [args.only] if args.only else list(MODEL_KEYS)
    runs: dict[str, dict[str, float]] = {}
    for model_key in model_keys:
        result = run(model_key, config)
        runs[f"baseline_{model_key}"] = result.test_metrics

    if len(runs) > 1:
        markdown, latex = reporting.build_comparison(runs)
        console.header("comparação entre baselines")
        print(markdown)
        (config.output_dir / "comparison_baselines.md").write_text(markdown, encoding="utf-8")
        (config.output_dir / "comparison_baselines.tex").write_text(latex, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
