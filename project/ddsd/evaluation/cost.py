"""Custo dos classificadores (`ddsd eval custo`): parâmetros, tamanho, treino, memória e inferência.

Tudo sai dos artefatos e relatórios já gravados, sem treinar nem medir de novo:
- parâmetros: contados no cabeçalho do `model.safetensors` do melhor checkpoint (sem carregar os pesos);
- tamanho: o arquivo dos pesos (Transformers) ou o `modelo.joblib` (TF-IDF);
- treino: média ± desvio entre sementes do tempo, da velocidade e do pico de memória (`run_manifest.json`);
- inferência: o tempo por fala medido na comparação pareada do pipeline (mesmo caminho e mesmas threads
  do pipeline) e, para os Transformers, o benchmark (CPU com 1, 2 e 4 threads e MPS).
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np

from ddsd.classifiers.model import resolve_checkpoint
from ddsd.common.io import read_json
from ddsd.evaluation.runs import Run


def safetensors_params(path: Path) -> int:
    """Número de parâmetros pelo cabeçalho do safetensors (8 bytes de tamanho + JSON com as formas)."""
    with path.open("rb") as handle:
        (size,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(size))
    return int(sum(math.prod(t["shape"]) for name, t in header.items() if name != "__metadata__"))


def _mean_sd(values: list[float]) -> dict[str, float]:
    return {"media": float(np.mean(values)), "desvio": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}


def transformer_cost(seed_runs: list[Run], chosen: Run) -> dict[str, Any]:
    """Custo de uma família de Transformer: o modelo da semente escolhida e o treino das sementes."""
    manifest = chosen.manifest
    weights = resolve_checkpoint(chosen.path, manifest["result"].get("best_checkpoint")) / "model.safetensors"
    timing = [r.manifest["timing"] for r in seed_runs]
    return {
        "execucao": chosen.path.name,
        "checkpoint_base": manifest["model"]["checkpoint"],
        "parametros": safetensors_params(weights),
        "tamanho_mb": weights.stat().st_size / 2**20,
        "precisao_treino": manifest["training"].get("precision_effective"),
        "sementes": len(seed_runs),
        "treino_min": _mean_sd([t["train_runtime_s"] / 60 for t in timing]),
        "passos_por_s": _mean_sd([t["steps_per_second"] for t in timing]),
        "pico_memoria_gb": _mean_sd([t["peak_memory_gb"] for t in timing]),
    }


def baseline_cost(run_dir: Path) -> dict[str, Any]:
    result = read_json(run_dir / "run_manifest.json")["result"]
    model_file = run_dir / result["model_file"]
    return {"execucao": run_dir.name, "vocabulario": result["vocabulary"], "tamanho_mb": model_file.stat().st_size / 2**20,
            "treino_s": result["wall_time_s"], "c": result["best_c"]}


def inference_from_reports(comparison_json: Path | None, benchmark_json: Path | None) -> dict[str, dict[str, Any]]:
    """Tempo de inferência por classificador: no sistema ao vivo (`comparacao.json`) e no benchmark (se houver)."""
    out: dict[str, dict[str, Any]] = {}
    if comparison_json is not None and comparison_json.exists():
        for name, t in read_json(comparison_json)["inferencia_ms"].items():
            out.setdefault(name, {})["pipeline_p50_ms"] = t["p50_ms"]
            out[name]["pipeline_p95_ms"] = t["p95_ms"]
    if benchmark_json is not None and benchmark_json.exists():
        for config, block in read_json(benchmark_json).get("classificadores", {}).items():
            for name, stats in block["isolado"].items():
                out.setdefault(name, {})[f"benchmark_{config}_p50_ms"] = 1000 * stats["p50_s"]
    return out
