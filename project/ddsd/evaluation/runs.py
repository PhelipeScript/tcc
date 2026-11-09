"""Execuções treinadas: descoberta, escolha da semente (pela VALIDAÇÃO) e inferência em textos arbitrários."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ddsd.common.io import read_json
from ddsd.common.paths import MODELS_DIR
from ddsd.training.config import TRAIN_ORDER


@dataclass(frozen=True)
class Run:
    condition: str  # "principal" | "controle" | "baseline"
    model: str
    seed: int | None
    path: Path

    @property
    def manifest(self) -> dict[str, Any]:
        return read_json(self.path / "run_manifest.json")

    @property
    def val_f1(self) -> float:
        return float(self.manifest["result"]["val_metrics"]["f1_macro"])

    @property
    def test_metrics(self) -> dict[str, float]:
        return self.manifest["result"]["test_metrics"]

    @property
    def label(self) -> str:
        seed = f"-s{self.seed}" if self.seed is not None else ""
        prefix = "controle/" if self.condition == "controle" else ""
        return f"{prefix}{self.model}{seed}"


def discover(models_dir: Path = MODELS_DIR) -> list[Run]:
    runs: list[Run] = []
    for condition, base in (("principal", models_dir), ("controle", models_dir / "controle")):
        for model in TRAIN_ORDER:
            for path in sorted(base.glob(f"{model}-s*")):
                if (path / "run_manifest.json").exists():
                    runs.append(Run(condition, model, int(path.name.split("-s")[-1]), path))
    for path in sorted(models_dir.glob("baseline_*")):
        if (path / "run_manifest.json").exists():
            runs.append(Run("baseline", path.name, None, path))
    return runs


def best_seed(runs: list[Run], condition: str, model: str) -> Run | None:
    """Semente escolhida pela F1 macro de **validação** (o teste nunca orienta escolhas)."""
    candidates = [r for r in runs if r.condition == condition and r.model == model]
    return max(candidates, key=lambda r: r.val_f1) if candidates else None


def seed_summary(runs: list[Run], condition: str, model: str, keys: tuple[str, ...]) -> dict[str, Any]:
    """Média e desvio padrão (amostral) das métricas de teste entre as sementes."""
    values = [r.test_metrics for r in runs if r.condition == condition and r.model == model]
    out: dict[str, Any] = {"n_sementes": len(values)}
    for key in keys:
        series = [v[key] for v in values]
        out[key] = {"media": statistics.fmean(series) if series else float("nan"),
                    "desvio": statistics.stdev(series) if len(series) > 1 else float("nan")}
    return out
