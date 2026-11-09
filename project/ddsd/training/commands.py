"""Subcomandos `ddsd train ...`."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from ddsd.observability.logger import configure_logging, get_logger

log = get_logger("treino")

#: Teste de fumaça: passos e amostras pequenos, sem early stopping.
SMOKE_STEPS = 60
SMOKE_TRAIN_SAMPLES = 4000
SMOKE_EVAL_SAMPLES = 1000


def _count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _training_config(args: argparse.Namespace):
    from ddsd.training.config import TrainingConfig

    config = TrainingConfig(seed=args.seed, precision=args.precision, allow_cpu=args.allow_cpu,
                            resume=args.resume, max_steps=args.max_steps,
                            max_train_samples=args.max_train_samples, max_eval_samples=args.max_eval_samples,
                            run_name=args.run_name, epochs_override=args.epochs)
    if args.data_dir:
        config.data_dir = args.data_dir.resolve()
    if args.output_dir:
        config.output_dir = args.output_dir.resolve()
    return config


def _estimate_full_run(result, config, model, full_train_rows: int, full_val_rows: int) -> dict[str, float]:
    """Projeção do tempo do treino completo a partir da velocidade medida no teste de fumaça."""
    manifest = json.loads((result.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    timing = manifest["timing"]
    steps_per_epoch = math.ceil(full_train_rows / model.effective_batch_size)
    epochs = config.epochs_override or model.epochs
    train_s = steps_per_epoch * epochs / max(result.steps_per_second, 1e-9)
    eval_rate = SMOKE_EVAL_SAMPLES / max(timing["val_eval_seconds"], 1e-9)
    eval_s = epochs * full_val_rows / eval_rate
    return {"steps_per_second": result.steps_per_second, "steps_per_epoch": steps_per_epoch, "epochs": epochs,
            "train_hours": round(train_s / 3600, 2), "eval_hours": round(eval_s / 3600, 2),
            "total_hours": round((train_s + eval_s) / 3600, 2), "peak_memory_gb": timing["peak_memory_gb"]}


def _release_accelerator_memory() -> None:
    """Libera o que sobrou da execução anterior (modelo, Trainer, cache do MPS/CUDA) antes da próxima."""
    import gc

    import torch

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cmd_train(args: argparse.Namespace) -> int:
    configure_logging(level="debug" if args.debug else "info")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from ddsd.common.io import write_json
    from ddsd.common.paths import MODELS_DIR
    from ddsd.training.config import MODELS, TRAIN_ORDER
    from ddsd.training.engine import run

    keys = list(TRAIN_ORDER) if args.model == "all" else [args.model]
    seeds = args.seeds or [args.seed]
    estimates: dict[str, dict[str, float]] = {}
    failures: list[str] = []
    for seed, key in [(seed, key) for seed in seeds for key in keys]:
        model = MODELS[key]
        args.seed = seed
        config = _training_config(args)
        if args.smoke:
            config.max_steps = config.max_steps or SMOKE_STEPS
            config.max_train_samples = config.max_train_samples or SMOKE_TRAIN_SAMPLES
            config.max_eval_samples = config.max_eval_samples or SMOKE_EVAL_SAMPLES
            config.early_stopping_patience = 0
            config.output_dir = MODELS_DIR / "smoke"
            config.run_name = config.run_name or f"{key}-{args.precision}"
        if args.skip_done and not args.smoke and (config.run_dir(model) / "run_manifest.json").exists():
            log.info("%s já concluído em %s; pulando", key, config.run_dir(model))
            continue
        if not args.smoke and (config.run_dir(model) / "checkpoints").exists():
            config.resume = True  # execução interrompida: retoma do último checkpoint de época
        try:
            result = run(model, config)
        except Exception as exc:  # noqa: BLE001 - um modelo que falha não impede os demais
            failures.append(f"{key}-s{seed}: {exc!r}")
            log.error("treino de %s (semente %d) falhou: %r", key, seed, exc)
            continue
        finally:
            _release_accelerator_memory()  # várias execuções seguidas no mesmo processo
        if args.smoke:
            full_rows = {n: _count_lines(config.data_dir / f"{n}.jsonl") for n in ("train", "val")}
            estimates[key] = _estimate_full_run(result, config, model, full_rows["train"], full_rows["val"])
            log.info("projeção do treino completo de %s: %s", key, estimates[key])
    if failures:
        log.error("execuções com falha: %s", failures)
    if args.smoke:
        total = round(sum(e["total_hours"] for e in estimates.values()), 2)
        write_json(MODELS_DIR / "smoke" / f"estimate-{args.precision}.json",
                   {"per_model": estimates, "total_hours_one_seed": total})
        log.info("projeção total (uma semente, %d modelo(s)): %.1f h", len(estimates), total)
    return 1 if failures else 0


def _cmd_compare(args: argparse.Namespace) -> int:
    configure_logging(level="info")
    from ddsd.common.io import read_json
    from ddsd.common.paths import MODELS_DIR, REPORTS_DIR
    from ddsd.training.reporting import comparison_tables

    runs = {}
    for manifest_path in sorted((args.models_dir or MODELS_DIR).glob("*/run_manifest.json")):
        runs[manifest_path.parent.name] = read_json(manifest_path)["result"]["test_metrics"]
    if not runs:
        log.error("nenhuma execução com run_manifest.json em %s", args.models_dir or MODELS_DIR)
        return 1
    markdown, latex = comparison_tables(runs)
    out = REPORTS_DIR / "training"
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.md").write_text("# Comparação no conjunto de teste\n\n" + markdown, encoding="utf-8")
    (out / "comparison.tex").write_text(latex, encoding="utf-8")
    print(markdown)
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    from ddsd.training.config import MODELS

    train = subparsers.add_parser("train", help="fine-tuning dos classificadores DD/NDD")
    train.add_argument("model", choices=[*sorted(MODELS), "all", "compare"], help="modelo, 'all' ou 'compare'")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--seeds", type=int, nargs="+", default=None, help="várias sementes em sequência")
    train.add_argument("--skip-done", action="store_true", help="pula execuções que já têm run_manifest.json")
    train.add_argument("--smoke", action="store_true",
                       help=f"teste de fumaça ({SMOKE_STEPS} passos) com projeção do tempo do treino completo")
    train.add_argument("--resume", action="store_true", help="retoma do último checkpoint da execução")
    train.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    train.add_argument("--data-dir", type=Path, default=None,
                       help="diretório com train/val/test.jsonl (padrão: corpus congelado)")
    train.add_argument("--output-dir", type=Path, default=None, help="raiz das execuções (padrão: models/)")
    train.add_argument("--run-name", default=None, help="nome da execução (padrão: <modelo>-s<semente>)")
    train.add_argument("--epochs", type=int, default=None, help="sobrescreve as épocas do modelo")
    train.add_argument("--max-steps", type=int, default=None)
    train.add_argument("--max-train-samples", type=int, default=None)
    train.add_argument("--max-eval-samples", type=int, default=None)
    train.add_argument("--allow-cpu", action="store_true", help="permite CPU (só para teste)")
    train.add_argument("--models-dir", type=Path, default=None, help="(compare) diretório das execuções")
    train.add_argument("--debug", action="store_true")
    train.set_defaults(handler=lambda a: _cmd_compare(a) if a.model == "compare" else _cmd_train(a))
