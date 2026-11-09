"""Núcleo de treino e avaliação (único módulo que toca a API de treino do `transformers`).

Ordem rígida, que sustenta a validade metodológica:

    hardware → semente → dados → modelo → treino
    → avalia VALIDAÇÃO → escolhe o limiar na VALIDAÇÃO
    → avalia TESTE uma única vez, com o limiar congelado
    → grava artefatos
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from transformers import EarlyStoppingCallback, Trainer, TrainerCallback, TrainingArguments

from ddsd.common.io import file_sha256, read_json, write_json
from ddsd.common.system import code_version, power_state
from ddsd.observability.logger import get_logger
from ddsd.training import data as data_module
from ddsd.training import hardware, modeling, reporting
from ddsd.training.config import LABEL_NAMES, ModelConfig, TrainingConfig
from ddsd.training.metrics import (apply_threshold, best_threshold, build_compute_metrics,
                                   classification_metrics, positive_scores)
from ddsd.training.seeding import set_all_seeds

log = get_logger("treino")

CODE_AT_IMPORT = code_version()


@dataclass
class RunResult:
    model_name: str
    run_dir: Path
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    decision_threshold: float
    best_checkpoint: str | None
    wall_time_s: float
    steps_per_second: float

    def to_dict(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "run_dir": str(self.run_dir), "val_metrics": self.val_metrics,
                "test_metrics": self.test_metrics, "decision_threshold": self.decision_threshold,
                # Relativo ao diretório da execução, para os artefatos funcionarem em qualquer SO.
                "best_checkpoint": self.best_checkpoint, "wall_time_s": round(self.wall_time_s, 1),
                "steps_per_second": round(self.steps_per_second, 3)}


class ProgressCallback(TrainerCallback):
    """Registra o progresso no logger e em `progress.json` a cada `logging_steps`."""

    def __init__(self, run_dir: Path, device: hardware.DeviceInfo) -> None:
        self.path = run_dir / "progress.json"
        self.device = device
        self.started = time.perf_counter()
        self.step_times: list[tuple[int, float]] = []
        self.peak_memory_gb = 0.0

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001 - assinatura do transformers
        now = time.perf_counter()
        self.step_times.append((state.global_step, now))
        if state.global_step % 10 == 0:
            self.peak_memory_gb = max(self.peak_memory_gb, hardware.allocated_memory_gb(self.device))

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        logs = logs or {}
        speed = self.steps_per_second()
        remaining = max(state.max_steps - state.global_step, 0)
        eta_min = remaining / speed / 60 if speed > 0 else float("nan")
        payload = {"step": state.global_step, "max_steps": state.max_steps, "epoch": round(state.epoch or 0, 3),
                   "logs": logs, "steps_per_second": round(speed, 3), "eta_minutes": round(eta_min, 1),
                   "elapsed_minutes": round((time.perf_counter() - self.started) / 60, 1),
                   "memory_gb": round(hardware.allocated_memory_gb(self.device), 2),
                   "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        write_json(self.path, payload)
        if "loss" in logs or "eval_loss" in logs:
            log.info("passo %d/%d época %.2f | %s | %.2f passos/s | ETA %.0f min", state.global_step,
                     state.max_steps, state.epoch or 0,
                     ", ".join(f"{k}={v:.4g}" for k, v in logs.items() if isinstance(v, float)), speed, eta_min)

    def steps_per_second(self, skip: int = 10) -> float:
        """Velocidade em regime, ignorando os primeiros passos (compilação de kernels no MPS)."""
        times = self.step_times[skip:] if len(self.step_times) > skip + 5 else self.step_times
        if len(times) < 2:
            return 0.0
        (s0, t0), (s1, t1) = times[0], times[-1]
        return (s1 - s0) / (t1 - t0) if t1 > t0 else 0.0


def build_training_arguments(model: ModelConfig, config: TrainingConfig, run_dir: Path, precision: str,
                             device: hardware.DeviceInfo) -> TrainingArguments:
    fields = {f.name for f in dataclasses.fields(TrainingArguments)}
    epochs = config.epochs_override or model.epochs
    arguments: dict[str, Any] = {
        "output_dir": str(run_dir / "checkpoints"),
        "per_device_train_batch_size": model.batch_size,
        "per_device_eval_batch_size": model.eval_batch_size,
        "gradient_accumulation_steps": model.grad_accum_steps,
        "learning_rate": model.learning_rate,
        "num_train_epochs": epochs,
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
        "dataloader_pin_memory": device.backend == "cuda",
        "report_to": [],
        "disable_tqdm": True,  # o progresso vai para o logger e para progress.json
        "bf16": precision == "bf16",
        "fp16": precision == "fp16",
        "label_names": ["labels"],
        "use_cpu": device.backend == "cpu",
    }
    # v5: `warmup_steps` aceita fração dos passos totais; versões antigas usam `warmup_ratio`.
    arguments["warmup_ratio" if "warmup_ratio" in fields else "warmup_steps"] = model.warmup_ratio
    if config.max_steps is not None:
        arguments["max_steps"] = config.max_steps
    unknown = sorted(set(arguments) - fields)
    if unknown:
        log.warn("TrainingArguments desta versão ignora %s", unknown)
    return TrainingArguments(**{k: v for k, v in arguments.items() if k in fields})


def load_deployed_model(trainer: Trainer, run_dir: Path) -> Any:
    """O modelo **exatamente como o pipeline o carrega**: o melhor checkpoint lido do disco, em fp32.

    Não basta usar `trainer.model`: com precisão mista, o Accelerate envolve o `forward` num
    `autocast` (bf16), e a avaliação herdaria esse envoltório sem aviso. Ler o checkpoint do disco
    garante que limiar e métricas sejam os do artefato que vai para produção.
    """
    import torch
    from transformers import AutoModelForSequenceClassification

    best = trainer.state.best_model_checkpoint
    if best is None:
        checkpoints = sorted((run_dir / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        if not checkpoints:
            raise RuntimeError("nenhum checkpoint gravado para avaliar")
        best = str(checkpoints[-1])
    device = next(trainer.model.parameters()).device
    model = AutoModelForSequenceClassification.from_pretrained(best, dtype=torch.float32)
    return model.to(device).eval()


def predict_fp32(model: Any, collator: Any, dataset: Any, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Logits em fp32, sem autocast (mesmo caminho numérico do pipeline em tempo real)."""
    import torch

    device = next(model.parameters()).device
    logits_parts, labels_parts = [], []
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            batch = collator([dataset[i] for i in range(start, min(start + batch_size, len(dataset)))])
            labels_parts.append(batch.pop("labels").numpy())
            batch = {k: v.to(device) for k, v in batch.items()}
            logits_parts.append(model(**batch).logits.float().cpu().numpy())
    return np.concatenate(logits_parts), np.concatenate(labels_parts)


def evaluate_split(model: Any, collator: Any, dataset: Any, batch_size: int, threshold: float | None = None):
    """(métricas, y_true, y_pred, y_score) de uma partição, em fp32; `threshold=None` usa argmax."""
    logits, y_true = predict_fp32(model, collator, dataset, batch_size)
    y_score = positive_scores(logits)
    y_pred = logits.argmax(axis=-1) if threshold is None else apply_threshold(y_score, threshold)
    return classification_metrics(y_true, y_pred, y_score), y_true, y_pred, y_score


def _corpus_provenance(data_dir: Path) -> dict[str, Any]:
    """SHA-256 das partições usadas e o resumo do build do corpus (se houver manifesto)."""
    files = {name: file_sha256(data_dir / f) for name, f in data_module.SPLIT_FILES.items()}
    manifest_path = data_dir.parent / "manifest.json"
    build = {}
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        build = {"built_at": manifest.get("built_at"), "config_sha256": manifest.get("config_sha256"),
                 "split": manifest.get("split"), "balance_mode": manifest.get("balance_mode")}
    return {"data_dir": str(data_dir), "split_sha256": files, "build": build}


def _code_drift() -> dict[str, Any]:
    """Registra o estado atual do repositório quando ele difere do código carregado (só informativo)."""
    now = code_version()
    keys = ("commit", "arquivos_alterados")
    return {} if all(now.get(k) == CODE_AT_IMPORT.get(k) for k in keys) else {"codigo_no_repositorio_ao_fim": now}


def run(model: ModelConfig, config: TrainingConfig) -> RunResult:
    """Uma sessão completa de treino e avaliação de um modelo."""
    started = time.perf_counter()
    run_dir = config.run_dir(model)
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("treino %s → %s", model.name, run_dir)

    device = hardware.describe_device()
    hardware.require_accelerator(device, config.allow_cpu)
    precision = hardware.resolve_precision(config.precision, device)
    precision_reason = "padrão (bf16 misto quando suportado pelo dispositivo, senão fp32)"
    if config.precision == "auto" and device.backend in model.precision_by_backend:
        precision, precision_reason = model.precision_by_backend[device.backend]
    hardware.configure_backend(device)
    if config.precision != "auto":
        precision_reason = "definida na linha de comando"
    log.info("dispositivo: %s (%s) | precisão efetiva: %s (%s)", device.name, device.backend, precision,
             precision_reason)
    power_start = power_state()
    if power_start.get("fonte_energia") == "bateria":
        log.warn("treino iniciado na bateria (%s%%): ligue o carregador; a velocidade e a duração ficam "
                 "comprometidas", power_start.get("bateria_pct"))
    set_all_seeds(config.seed)

    dataset = data_module.load_splits(config.data_dir, config.text_field)
    if config.max_train_samples is not None:
        dataset["train"] = data_module.subsample(dataset["train"], config.max_train_samples, config.seed)
    if config.max_eval_samples is not None:
        for split in ("val", "test"):
            dataset[split] = data_module.subsample(dataset[split], config.max_eval_samples, config.seed)
    summary = data_module.describe_splits(dataset)
    log.info("partições: %s", {name: info["n"] for name, info in summary.items()})
    test_meta = data_module.metadata(dataset["test"])

    tokenizer = data_module.build_tokenizer(model)
    tokenized = data_module.tokenize_splits(dataset, tokenizer, model.max_length)
    lengths = data_module.token_length_report(tokenized, model.max_length)
    log.info("subtokens: média %.1f, p99 %.0f, máx %d, truncados %.3f%%", lengths["media"], lengths["p99"],
             lengths["max"], 100 * lengths["fracao_truncada"])

    module = modeling.build_model(model)
    params = modeling.count_parameters(module)
    log.info("%s: %.1fM parâmetros | lote %d×%d | lr %g | épocas %d", model.checkpoint, params["total"] / 1e6,
             model.batch_size, model.grad_accum_steps, model.learning_rate, config.epochs_override or model.epochs)

    arguments = build_training_arguments(model, config, run_dir, precision, device)
    progress = ProgressCallback(run_dir, device)
    callbacks: list[TrainerCallback] = [progress]
    if config.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience))
    trainer = Trainer(model=module, args=arguments, train_dataset=tokenized["train"],
                      eval_dataset=tokenized["val"], processing_class=tokenizer,
                      data_collator=data_module.build_collator(tokenizer),
                      compute_metrics=build_compute_metrics(), callbacks=callbacks)

    checkpoints = sorted((run_dir / "checkpoints").glob("checkpoint-*"))
    resume = bool(config.resume and checkpoints)
    if config.resume and not checkpoints:
        log.warn("--resume pedido, mas não há checkpoints em %s; começando do zero", run_dir / "checkpoints")
    train_output = trainer.train(resume_from_checkpoint=True if resume else None)
    speed = progress.steps_per_second()
    log.info("treino concluído em %.0f s (%.2f passos/s em regime, pico de memória %.2f GiB)",
             train_output.metrics.get("train_runtime", 0.0), speed, progress.peak_memory_gb)

    eval_started = time.perf_counter()
    deployed = load_deployed_model(trainer, run_dir)
    collator = trainer.data_collator
    val_metrics, val_true, _, val_score = evaluate_split(deployed, collator, tokenized["val"], model.eval_batch_size)
    eval_seconds = time.perf_counter() - eval_started
    threshold = 0.5
    if config.tune_threshold:
        threshold, value = best_threshold(val_true, val_score, config.metric_for_best_model)
        log.info("limiar escolhido na validação: %.3f (%s = %.4f)", threshold, config.metric_for_best_model, value)

    # Teste: uma única vez, com o limiar congelado.
    test_metrics, test_true, test_pred, test_score = evaluate_split(deployed, collator, tokenized["test"],
                                                                    model.eval_batch_size, threshold)
    log.info("TESTE (limiar %.3f): acc %.4f | f1_macro %.4f | EER %.4f | AUC %.4f", threshold,
             test_metrics["accuracy"], test_metrics["f1_macro"], test_metrics["eer"], test_metrics["roc_auc"])

    write_json(run_dir / "metrics_val.json", val_metrics)
    write_json(run_dir / "metrics_test.json", test_metrics)
    reporting.save_confusion_matrix(test_true, test_pred, run_dir, "test")
    reporting.save_curves(test_true, test_score, run_dir, "test", test_metrics["eer"])
    reporting.save_predictions(test_true, test_pred, test_score, test_meta, run_dir, "test")
    reporting.save_grouped_metrics(test_true, test_pred, test_score, test_meta, run_dir, "test")
    reporting.save_error_samples(test_meta, test_true, test_pred, test_score, run_dir, "test")

    best = trainer.state.best_model_checkpoint
    result = RunResult(
        model_name=model.name, run_dir=run_dir, val_metrics=val_metrics, test_metrics=test_metrics,
        decision_threshold=threshold,
        best_checkpoint=str(Path(best).relative_to(run_dir)) if best else None,
        wall_time_s=time.perf_counter() - started, steps_per_second=speed,
    )
    steps_per_epoch = math.ceil(summary["train"]["n"] / model.effective_batch_size)
    reporting.save_run_manifest({
        "model": dataclasses.asdict(model),
        "training": {**{k: (str(v) if isinstance(v, Path) else v) for k, v in dataclasses.asdict(config).items()},
                     "precision_effective": precision, "precision_reason": precision_reason, "resumed": resume,
                     "final_evaluation": "melhor checkpoint lido do disco, fp32 sem autocast (igual ao pipeline)"},
        "hardware": device.to_dict(),
        "energia": {"inicio": power_start, "fim": power_state()},
        "codigo": CODE_AT_IMPORT,
        **_code_drift(),
        "parameters": params,
        "token_lengths": lengths,
        "splits": summary,
        "corpus": _corpus_provenance(config.data_dir),
        "timing": {"steps_per_epoch": steps_per_epoch, "steps_per_second": round(speed, 3),
                   "val_eval_seconds": round(eval_seconds, 1), "peak_memory_gb": round(progress.peak_memory_gb, 2),
                   "train_runtime_s": train_output.metrics.get("train_runtime")},
        "label_names": list(LABEL_NAMES),
        "result": result.to_dict(),
    }, run_dir)
    log.info("artefatos gravados em %s", run_dir)
    return result
