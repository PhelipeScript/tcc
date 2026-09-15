"""
Descrição:
    Interface de linha de comando compartilhada pelos scripts de treino.

    Os três modelos aceitam exatamente as mesmas opções; o que muda são os
    valores padrão, que vêm do `ModelConfig` de cada um. Concentrar o parser
    aqui evita repetir a declaração de vinte argumentos em três arquivos.

    Qualquer opção não informada na linha de comando mantém o padrão do modelo,
    de modo que `python -m src.training.train_bertimbau` sem argumentos já roda
    a configuração definida na metodologia do TCC.

Uso:
    from src.training.cli import main_for_model
    raise SystemExit(main_for_model("bertimbau"))
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from src.training.config import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    MODELS,
    ModelConfig,
    TrainingConfig,
)
from src.training.engine import RunResult, run


def build_parser(model_key: str) -> argparse.ArgumentParser:
    """
    Descrição:
        Monta o parser de argumentos para um modelo, usando sua configuração
        como fonte dos valores padrão.

    Args:
        model_key: Chave do modelo em `config.MODELS`.

    Retorna:
        Parser configurado.

    Levanta:
        KeyError: Se a chave do modelo não existir.
    """
    model = MODELS[model_key]
    defaults = TrainingConfig()

    parser = argparse.ArgumentParser(
        prog=f"train_{model_key}",
        description=(
            f"Fine-tuning de {model.name} ({model.checkpoint}) "
            "para classificação DD vs NDD."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    grupo_modelo = parser.add_argument_group("modelo e hiperparâmetros")
    grupo_modelo.add_argument(
        "--checkpoint", default=model.checkpoint, help="identificador do modelo no Hugging Face Hub"
    )
    grupo_modelo.add_argument("--epochs", type=int, default=model.epochs, help="número de épocas")
    grupo_modelo.add_argument(
        "--batch-size", type=int, default=model.batch_size, help="lote por dispositivo"
    )
    grupo_modelo.add_argument(
        "--grad-accum-steps",
        type=int,
        default=model.grad_accum_steps,
        help="passos de acumulação de gradiente (lote efetivo = lote x acumulação)",
    )
    grupo_modelo.add_argument(
        "--eval-batch-size", type=int, default=model.eval_batch_size, help="lote na avaliação"
    )
    grupo_modelo.add_argument(
        "--lr", type=float, default=model.learning_rate, help="taxa de aprendizado de pico"
    )
    grupo_modelo.add_argument(
        "--max-length", type=int, default=model.max_length, help="comprimento máximo em subtokens"
    )
    grupo_modelo.add_argument(
        "--warmup-ratio", type=float, default=model.warmup_ratio, help="fração de passos em warmup"
    )
    grupo_modelo.add_argument(
        "--dropout", type=float, default=model.dropout, help="dropout do cabeçalho de classificação"
    )

    grupo_dados = parser.add_argument_group("dados")
    grupo_dados.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="diretório com as partições"
    )
    grupo_dados.add_argument(
        "--text-field",
        choices=("text", "text_raw"),
        default=defaults.text_field,
        help="campo de entrada: `text` (normalizado) ou `text_raw` (com acento e pontuação)",
    )

    grupo_exec = parser.add_argument_group("execução")
    grupo_exec.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="diretório raiz das execuções"
    )
    grupo_exec.add_argument(
        "--run-name", default=None, help="nome do subdiretório da execução (padrão: nome do modelo)"
    )
    grupo_exec.add_argument("--seed", type=int, default=defaults.seed, help="semente global")
    grupo_exec.add_argument(
        "--precision",
        choices=("bf16", "fp16", "fp32"),
        default=defaults.precision,
        help="precisão numérica; rebaixada automaticamente se a GPU não suportar",
    )
    grupo_exec.add_argument(
        "--early-stopping-patience",
        type=int,
        default=defaults.early_stopping_patience,
        help="épocas sem melhora antes de parar (0 desliga)",
    )
    grupo_exec.add_argument(
        "--metric-for-best-model",
        default=defaults.metric_for_best_model,
        help="métrica de seleção do melhor checkpoint",
    )
    grupo_exec.add_argument(
        "--num-workers",
        type=int,
        default=defaults.dataloader_num_workers,
        help="processos do DataLoader (0 é o mais seguro no Windows)",
    )
    grupo_exec.add_argument(
        "--no-tune-threshold",
        action="store_true",
        help="não ajusta o limiar de decisão na validação (mantém argmax em 0,5)",
    )

    grupo_teste = parser.add_argument_group("teste de fumaça")
    grupo_teste.add_argument(
        "--allow-cpu",
        action="store_true",
        help="permite rodar sem CUDA; serve apenas para validar o código, não para treinar",
    )
    grupo_teste.add_argument(
        "--max-train-samples", type=int, default=None, help="limita o treino a N exemplos"
    )
    grupo_teste.add_argument(
        "--max-eval-samples", type=int, default=None, help="limita validação e teste a N exemplos"
    )
    grupo_teste.add_argument(
        "--max-steps", type=int, default=None, help="limita o número de passos de otimização"
    )

    return parser


def configs_from_args(model_key: str, args: argparse.Namespace) -> tuple[ModelConfig, TrainingConfig]:
    """
    Descrição:
        Converte os argumentos da linha de comando nas duas configurações do
        treino, sobrepondo os padrões do modelo.

    Args:
        model_key: Chave do modelo em `config.MODELS`.
        args: Namespace produzido pelo parser.

    Retorna:
        Tupla `(model_config, training_config)`.
    """
    model = dataclasses.replace(
        MODELS[model_key],
        checkpoint=args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        warmup_ratio=args.warmup_ratio,
        dropout=args.dropout,
    )

    training = TrainingConfig(
        seed=args.seed,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        text_field=args.text_field,
        precision=args.precision,
        allow_cpu=args.allow_cpu,
        early_stopping_patience=args.early_stopping_patience,
        metric_for_best_model=args.metric_for_best_model,
        dataloader_num_workers=args.num_workers,
        tune_threshold=not args.no_tune_threshold,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_steps=args.max_steps,
        run_name=args.run_name,
    )
    return model, training


def main_for_model(model_key: str, argv: list[str] | None = None) -> int:
    """
    Descrição:
        Ponto de entrada compartilhado pelos três scripts de treino: interpreta
        os argumentos, monta as configurações e executa o treino.

    Args:
        model_key: Chave do modelo em `config.MODELS`.
        argv: Argumentos de linha de comando. `None` usa `sys.argv`.

    Retorna:
        Código de saída do processo (0 em sucesso).
    """
    parser = build_parser(model_key)
    args = parser.parse_args(argv)
    model, training = configs_from_args(model_key, args)
    result: RunResult = run(model, training)
    return 0 if result.test_metrics else 1
