"""
Descrição:
    Treina os três modelos em sequência, sobre o mesmo dataset, e gera a tabela
    comparativa final.

    Todos recebem o mesmo lote efetivo (32), o mesmo `max_length` (128), o mesmo
    warmup (10%), o mesmo dropout (0,1) e a mesma semente — de modo que a
    diferença observada entre eles seja atribuível ao modelo, e não à
    configuração de treino.

    Se um modelo falhar, os demais continuam: a falha é registrada e reportada
    no fim, em vez de abortar a bateria inteira depois de horas de GPU.

Uso:
    python -m src.training.train_all
    python -m src.training.train_all --models bertimbau debertinha
    python -m src.training.train_all --text-field text_raw --run-suffix _raw

Saída:
    outputs/<modelo>/...            (artefatos de cada execução)
    outputs/comparison.md           (tabela comparativa em Markdown)
    outputs/comparison.tex          (mesma tabela em LaTeX, para o TCC)
    outputs/comparison.json         (resultados consolidados)
"""

from __future__ import annotations

import argparse
import dataclasses
import traceback
from pathlib import Path

from src.data_prep import console
from src.training import reporting
from src.training.config import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    MODELS,
    TrainingConfig,
)
from src.training.engine import RunResult, run

#: Ordem de treino: do menor para o maior modelo.
#:
#: Treinar o DeBERTinha primeiro é deliberado — é o mais rápido, então qualquer
#: erro de configuração aparece em minutos, e não depois de horas no maior.
DEFAULT_ORDER: tuple[str, ...] = ("debertinha", "bertimbau", "albertina")


def build_parser() -> argparse.ArgumentParser:
    """
    Descrição:
        Monta o parser de argumentos da bateria de treinos.

    Retorna:
        Parser configurado.
    """
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(
        prog="train_all",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODELS),
        default=list(DEFAULT_ORDER),
        help="modelos a treinar, na ordem informada",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="diretório das partições")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="diretório raiz das execuções")
    parser.add_argument(
        "--text-field",
        choices=("text", "text_raw"),
        default=defaults.text_field,
        help="campo de entrada dos três modelos",
    )
    parser.add_argument(
        "--run-suffix",
        default="",
        help="sufixo do nome de cada execução, para não sobrescrever uma bateria anterior",
    )
    parser.add_argument("--seed", type=int, default=defaults.seed, help="semente global")
    parser.add_argument(
        "--precision", choices=("bf16", "fp16", "fp32"), default=defaults.precision, help="precisão numérica"
    )
    parser.add_argument("--allow-cpu", action="store_true", help="permite rodar sem CUDA (teste de fumaça)")
    parser.add_argument("--max-train-samples", type=int, default=None, help="limita o treino a N exemplos")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="limita avaliação a N exemplos")
    parser.add_argument("--max-steps", type=int, default=None, help="limita os passos de otimização")
    return parser


def train_one(model_key: str, args: argparse.Namespace) -> RunResult:
    """
    Descrição:
        Treina um único modelo com a configuração comum da bateria.

    Args:
        model_key: Chave do modelo em `config.MODELS`.
        args: Argumentos da bateria.

    Retorna:
        Resultado da execução.
    """
    model = MODELS[model_key]
    training = TrainingConfig(
        seed=args.seed,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        text_field=args.text_field,
        precision=args.precision,
        allow_cpu=args.allow_cpu,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_steps=args.max_steps,
        run_name=f"{model.name}{args.run_suffix}" if args.run_suffix else None,
    )
    return run(model, training)


def write_comparison(
    results: dict[str, RunResult], failures: dict[str, str], output_dir: Path
) -> list[Path]:
    """
    Descrição:
        Grava a tabela comparativa em Markdown, LaTeX e JSON.

    Args:
        results: Execuções concluídas, por nome de modelo.
        failures: Modelos que falharam, com a mensagem de erro.
        output_dir: Diretório raiz das execuções.

    Retorna:
        Lista dos caminhos gravados.
    """
    metrics = {name: result.test_metrics for name, result in results.items()}
    markdown, latex = reporting.build_comparison(metrics)

    if failures:
        markdown += "\n## Falhas\n\n" + "\n".join(
            f"- **{name}**: {message}" for name, message in failures.items()
        ) + "\n"

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "comparison.md",
        output_dir / "comparison.tex",
    ]
    paths[0].write_text(markdown, encoding="utf-8")
    paths[1].write_text(latex, encoding="utf-8")
    paths.append(
        reporting.save_json(
            {
                "results": {name: result.to_dict() for name, result in results.items()},
                "failures": failures,
            },
            output_dir / "comparison.json",
        )
    )
    return paths


def main(argv: list[str] | None = None) -> int:
    """
    Descrição:
        Executa a bateria completa: treina os modelos pedidos, um após o outro,
        e grava a comparação final.

    Args:
        argv: Argumentos de linha de comando. `None` usa `sys.argv`.

    Retorna:
        0 se todos os modelos treinaram; 1 se algum falhou.
    """
    args = build_parser().parse_args(argv)

    console.header("Bateria de treinos")
    console.info(f"modelos: {', '.join(args.models)}")
    console.info(f"campo de entrada: {args.text_field}   semente: {args.seed}")

    results: dict[str, RunResult] = {}
    failures: dict[str, str] = {}

    for model_key in args.models:
        try:
            results[model_key] = train_one(model_key, args)
        except Exception as exc:  # noqa: BLE001 - um modelo não deve derrubar a bateria
            failures[model_key] = f"{type(exc).__name__}: {exc}"
            console.error(f"{model_key} falhou: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    console.header("Comparação final (conjunto de teste)")
    if results:
        console.table(
            ["modelo", *reporting.REPORT_METRICS, "tempo(s)", "VRAM(GiB)"],
            [
                [
                    name,
                    *[f"{result.test_metrics.get(key, float('nan')):.4f}" for key in reporting.REPORT_METRICS],
                    f"{result.wall_time_s:.0f}",
                    f"{result.peak_vram_gb:.2f}",
                ]
                for name, result in results.items()
            ],
        )
    for path in write_comparison(results, failures, args.output_dir):
        console.ok(str(path))

    if failures:
        console.error(f"{len(failures)} modelo(s) falharam: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
