"""
Descrição:
    Núcleo compartilhado de treino e avaliação.

    Este é o único módulo que toca a API do `transformers`, e é o que os três
    scripts de modelo chamam. Concentrar aqui a construção de `TrainingArguments`
    e do `Trainer` significa que uma futura mudança de versão da biblioteca é
    uma edição em um arquivo, não em três.

    O fluxo de `run()` é deliberadamente rígido quanto à disciplina experimental:

        hardware -> semente -> dados -> modelo -> treino
        -> avalia VALIDAÇÃO -> escolhe o limiar na VALIDAÇÃO
        -> avalia TESTE uma única vez, com o limiar congelado
        -> grava artefatos

    O conjunto de teste é tocado exatamente uma vez, no fim. O limiar de decisão
    é escolhido apenas na validação. Qualquer desvio disso invalidaria a
    afirmação de que o teste ficou reservado durante todo o desenvolvimento.

    Compatibilidade de versão: o transformers v5 renomeou e removeu campos de
    `TrainingArguments` (`evaluation_strategy` virou `eval_strategy`,
    `warmup_ratio` foi absorvido por `warmup_steps`, que agora aceita float como
    proporção). `build_training_arguments` monta os argumentos por nomes
    canônicos e os traduz para o que a versão instalada realmente aceita,
    inspecionando a dataclass. Assim o código roda tanto na v5 quanto na v4.

Uso:
    from src.training.engine import run
    result = run(MODELS["bertimbau"], TrainingConfig())
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from src.data_prep import console
from src.training import data as data_module
from src.training import hardware, modeling, reporting, seeding
from src.training.config import LABEL_NAMES, ModelConfig, TrainingConfig
from src.training.metrics import (
    apply_threshold,
    best_threshold,
    build_compute_metrics,
    classification_metrics,
    positive_scores,
)

#: Tradução dos nomes canônicos para os nomes aceitos em cada versão do
#: transformers. A primeira alternativa presente na dataclass é a usada.
_ARG_ALIASES: dict[str, tuple[str, ...]] = {
    "eval_strategy": ("eval_strategy", "evaluation_strategy"),
}


@dataclass
class RunResult:
    """
    Descrição:
        Resultado consolidado de uma execução de treino.

    Atributos:
        model_name: Nome curto do modelo treinado.
        run_dir: Diretório com os artefatos da execução.
        val_metrics: Métricas na validação, com o limiar padrão de argmax.
        test_metrics: Métricas no teste, com o limiar escolhido na validação.
        decision_threshold: Limiar aplicado ao teste.
        best_checkpoint: Caminho do melhor checkpoint, se houver.
        wall_time_s: Tempo total da execução, em segundos.
        peak_vram_gb: Pico de VRAM durante o treino, em GiB.
    """

    model_name: str
    run_dir: Path
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    decision_threshold: float
    best_checkpoint: str | None
    wall_time_s: float
    peak_vram_gb: float

    def to_dict(self) -> dict[str, Any]:
        """
        Descrição:
            Serializa o resultado da execução.

        Retorna:
            Dicionário pronto para JSON.
        """
        return {
            "model_name": self.model_name,
            "run_dir": str(self.run_dir),
            "val_metrics": self.val_metrics,
            "test_metrics": self.test_metrics,
            "decision_threshold": self.decision_threshold,
            "best_checkpoint": self.best_checkpoint,
            "wall_time_s": round(self.wall_time_s, 1),
            "peak_vram_gb": round(self.peak_vram_gb, 2),
        }


def _supported_argument_names() -> set[str]:
    """
    Descrição:
        Lista os nomes de campo aceitos pela `TrainingArguments` instalada.

    Retorna:
        Conjunto de nomes de campo.
    """
    return {field.name for field in dataclasses.fields(TrainingArguments)}


def _resolve_arguments(canonical: dict[str, Any]) -> dict[str, Any]:
    """
    Descrição:
        Traduz um dicionário de argumentos canônicos para os nomes aceitos pela
        versão instalada do transformers, descartando com aviso os que não
        existem em versão alguma.

    Args:
        canonical: Argumentos por nome canônico.

    Retorna:
        Argumentos prontos para passar a `TrainingArguments`.
    """
    supported = _supported_argument_names()
    resolved: dict[str, Any] = {}

    for name, value in canonical.items():
        for candidate in _ARG_ALIASES.get(name, (name,)):
            if candidate in supported:
                resolved[candidate] = value
                break
        else:
            console.warn(
                f"TrainingArguments desta versão não aceita {name!r}; argumento ignorado"
            )
    return resolved


def build_training_arguments(
    model: ModelConfig, config: TrainingConfig, run_dir: Path, precision: str
) -> TrainingArguments:
    """
    Descrição:
        Monta os `TrainingArguments` da execução.

        O warmup é expresso como proporção: no transformers v5 `warmup_steps`
        aceita float em `[0, 1)` e o interpreta como fração dos passos totais,
        que é exatamente o "warmup de 10%" da metodologia. Em versões antigas o
        campo equivalente é `warmup_ratio`, tratado pela tradução de nomes.

    Args:
        model: Configuração do modelo.
        config: Configuração comum da execução.
        run_dir: Diretório de saída.
        precision: Precisão resolvida (`"bf16"`, `"fp16"` ou `"fp32"`).

    Retorna:
        `TrainingArguments` pronto para o `Trainer`.
    """
    canonical: dict[str, Any] = {
        "output_dir": str(run_dir / "checkpoints"),
        "per_device_train_batch_size": model.batch_size,
        "per_device_eval_batch_size": model.eval_batch_size,
        "gradient_accumulation_steps": model.grad_accum_steps,
        "learning_rate": model.learning_rate,
        "num_train_epochs": model.epochs,
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm,
        "lr_scheduler_type": config.lr_scheduler_type,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": config.save_total_limit,
        "load_best_model_at_end": config.early_stopping_patience > 0,
        "metric_for_best_model": config.metric_for_best_model,
        "greater_is_better": True,
        "logging_steps": 100,
        "seed": config.seed,
        "data_seed": config.seed,
        "dataloader_num_workers": config.dataloader_num_workers,
        "dataloader_pin_memory": True,
        "report_to": [],
        "bf16": precision == "bf16",
        "fp16": precision == "fp16",
        "label_names": ["labels"],
    }

    # Warmup como proporção dos passos totais.
    supported = _supported_argument_names()
    if "warmup_ratio" in supported:
        canonical["warmup_ratio"] = model.warmup_ratio
    else:
        canonical["warmup_steps"] = model.warmup_ratio

    if config.max_steps is not None:
        canonical["max_steps"] = config.max_steps

    return TrainingArguments(**_resolve_arguments(canonical))


def build_trainer(
    model: ModelConfig,
    config: TrainingConfig,
    module: Any,
    tokenizer: Any,
    dataset: Any,
    arguments: TrainingArguments,
) -> Trainer:
    """
    Descrição:
        Instancia o `Trainer` com colator de padding dinâmico, função de
        métricas e early stopping.

        Usa `processing_class` para passar o tokenizador: no transformers v5 o
        parâmetro `tokenizer` do `Trainer` deixou de existir.

    Args:
        model: Configuração do modelo.
        config: Configuração comum da execução.
        module: Modelo carregado.
        tokenizer: Tokenizador do modelo.
        dataset: `DatasetDict` tokenizado.
        arguments: `TrainingArguments` montado.

    Retorna:
        `Trainer` pronto para treinar.
    """
    callbacks = []
    if config.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)
        )

    return Trainer(
        model=module,
        args=arguments,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        processing_class=tokenizer,
        data_collator=data_module.build_collator(tokenizer),
        compute_metrics=build_compute_metrics(),
        callbacks=callbacks,
    )


def evaluate_split(
    trainer: Trainer, dataset: Any, threshold: float | None = None
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Descrição:
        Avalia uma partição e devolve métricas e vetores brutos.

    Args:
        trainer: `Trainer` treinado.
        dataset: Partição a avaliar.
        threshold: Limiar de decisão. `None` usa argmax (equivalente a 0,5).

    Retorna:
        Tupla `(métricas, y_true, y_pred, y_score)`.
    """
    output = trainer.predict(dataset)
    logits = output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    y_true = np.asarray(output.label_ids)
    y_score = positive_scores(logits)
    y_pred = (
        np.asarray(logits).argmax(axis=-1)
        if threshold is None
        else apply_threshold(y_score, threshold)
    )
    return classification_metrics(y_true, y_pred, y_score), y_true, y_pred, y_score


def run(model: ModelConfig, config: TrainingConfig) -> RunResult:
    """
    Descrição:
        Executa uma sessão completa de treino e avaliação de um modelo.

        Ordem das etapas e por quê:

        1. valida o hardware e resolve a precisão — falhar aqui é barato
        2. semeia todos os geradores aleatórios
        3. carrega e tokeniza os dados; reporta a distribuição de comprimento
        4. constrói o modelo com o dropout e o dtype corretos
        5. treina, selecionando o melhor checkpoint pela métrica de validação
        6. avalia a validação e escolhe nela o limiar de decisão
        7. avalia o teste **uma vez**, com o limiar congelado
        8. grava métricas, curvas, matriz de confusão, predições e manifest

    Args:
        model: Configuração do modelo a treinar.
        config: Configuração comum da execução.

    Retorna:
        `RunResult` com as métricas e os caminhos dos artefatos.

    Levanta:
        RuntimeError: Se CUDA não estiver disponível e `allow_cpu` for `False`.
        FileNotFoundError: Se as partições do dataset não existirem.
    """
    started = time.perf_counter()
    run_dir = config.run_dir(model)
    run_dir.mkdir(parents=True, exist_ok=True)

    console.header(f"Treino: {model.name}")
    console.info(f"saída: {run_dir}")

    # 1. Hardware.
    device = hardware.describe_device()
    hardware.log_device(device)
    hardware.require_cuda(allow_cpu=config.allow_cpu)
    precision = hardware.resolve_precision(config.precision, device)
    hardware.enable_tf32()
    console.info(f"precisão efetiva: {precision}")

    # 2. Reprodutibilidade.
    seeding.set_all_seeds(config.seed)

    # 3. Dados.
    dataset = data_module.load_splits(config.data_dir, config.text_field)
    if config.max_train_samples is not None:
        dataset["train"] = data_module.subsample(
            dataset["train"], config.max_train_samples, config.seed
        )
    if config.max_eval_samples is not None:
        for split in ("val", "test"):
            dataset[split] = data_module.subsample(
                dataset[split], config.max_eval_samples, config.seed
            )

    summary = data_module.describe_splits(dataset)
    data_module.log_splits(summary)
    console.info(f"campo de entrada: {config.text_field}")

    # Guarda os textos e provedores do teste antes da tokenização, que os remove.
    test_texts = list(dataset["test"]["text"])
    test_sources = list(dataset["test"]["source"])

    tokenizer = data_module.build_tokenizer(model)
    tokenized = data_module.tokenize_splits(dataset, tokenizer, model.max_length)
    lengths = data_module.token_length_report(tokenized, tokenizer, model.max_length)
    console.detail(
        f"comprimento em subtokens: media={lengths['media']:.1f} "
        f"p95={lengths['p95']:.0f} p99={lengths['p99']:.0f} max={lengths['max']} "
        f"(max_length={model.max_length}, truncados={100 * lengths['fracao_truncada']:.3f}%)"
    )

    # 4. Modelo.
    module = modeling.build_model(model)
    param_counts = modeling.log_model(model, module)

    # 5. Treino.
    arguments = build_training_arguments(model, config, run_dir, precision)
    trainer = build_trainer(model, config, module, tokenizer, tokenized, arguments)

    hardware.reset_peak_memory()
    console.header("Treinando")
    train_output = trainer.train()
    peak_vram = hardware.peak_memory_gb()
    console.ok(
        f"treino concluído: {train_output.metrics.get('train_runtime', 0):.0f}s, "
        f"pico de VRAM {peak_vram:.2f} GiB"
    )

    # 6. Validação e escolha do limiar.
    val_metrics, val_true, _, val_score = evaluate_split(trainer, tokenized["val"])
    reporting.log_metrics(val_metrics, "Validação (limiar 0,5)")

    threshold = 0.5
    if config.tune_threshold:
        threshold, value = best_threshold(val_true, val_score, config.metric_for_best_model)
        console.ok(
            f"limiar escolhido na validação: {threshold:.3f} "
            f"({config.metric_for_best_model}={value:.4f})"
        )

    # 7. Teste — uma única vez, com o limiar congelado.
    test_metrics, test_true, test_pred, test_score = evaluate_split(
        trainer, tokenized["test"], threshold
    )
    reporting.log_metrics(test_metrics, f"Teste (limiar {threshold:.3f})")

    # 8. Artefatos.
    reporting.save_metrics(val_metrics, run_dir, "val")
    reporting.save_metrics(test_metrics, run_dir, "test")
    reporting.save_confusion_matrix(test_true, test_pred, run_dir, "test")
    reporting.save_curves(test_true, test_score, run_dir, "test", test_metrics["eer"])
    reporting.save_predictions(test_true, test_pred, test_score, test_sources, run_dir, "test")
    reporting.save_per_source_metrics(
        test_true, test_pred, test_score, test_sources, run_dir, "test"
    )
    reporting.save_error_samples(test_texts, test_true, test_pred, test_score, run_dir, "test")

    elapsed = time.perf_counter() - started
    result = RunResult(
        model_name=model.name,
        run_dir=run_dir,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        decision_threshold=threshold,
        best_checkpoint=trainer.state.best_model_checkpoint,
        wall_time_s=elapsed,
        peak_vram_gb=peak_vram,
    )

    reporting.save_run_manifest(
        {
            "model": dataclasses.asdict(model),
            "training": {
                **{
                    key: (str(value) if isinstance(value, Path) else value)
                    for key, value in dataclasses.asdict(config).items()
                },
                "precision_effective": precision,
            },
            "hardware": device.to_dict(),
            "parameters": param_counts,
            "token_lengths": lengths,
            "splits": summary,
            "label_names": list(LABEL_NAMES),
            "result": result.to_dict(),
        },
        run_dir,
    )

    console.ok(f"artefatos gravados em {run_dir}")
    return result
