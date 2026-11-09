"""Fontes do corpus: importação congelada do corpus inicial e leitura de todas as fontes.

Layout em `data/corpus/sources/`:

    legacy/<provedor>/{dd,ndd}.jsonl         corpus inicial (copiado uma vez)
    legacy/MANIFEST.json                     commit de origem + SHA-256 de cada arquivo
    mitigation/<provedor>/dd*.jsonl          geração nova da estratégia `comprimento`
    mitigation/<provedor>/ndd*.jsonl         (vários arquivos por provedor são permitidos,
                                             um por processo gerador)

O nome do arquivo define a classe (`dd*` ou `ndd*`); o campo `label` de cada linha
precisa concordar com ele.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ddsd.common.io import file_sha256, read_json, read_jsonl, write_json
from ddsd.corpus.config import LABELS, ORIGINS, CorpusConfig, Label
from ddsd.observability.logger import get_logger

log = get_logger("corpus.fontes")

#: Modelo gerador de cada provedor do corpus inicial.
LEGACY_MODELS: dict[str, str] = {
    "glm": "zai-org/glm-4.6v-flash",
    "groq": "qwen/qwen3.8-27b",
    "lmstudio": "google/gemma-4-e4b",
    "mistral": "mistral-small-latest",
    "nvidia": "moonshotai/kimi-k3",
    "opencode": "opencode (CLI)",
    "openrouter": "openrouter/free",
    "qwen": "qwen/qwen3.6-35b-a3b",
    "sonnet": "claude-sonnet (CLI)",
}


@dataclass(frozen=True, slots=True)
class SourceRow:
    """Uma linha de fonte, com proveniência completa."""

    text_raw: str
    label: Label
    source: str
    origin: str
    strategy: str
    model: str
    file: str
    line: int
    extra: dict[str, Any]


def _label_of_file(path: Path) -> Label | None:
    name = path.name.lower()
    if name.startswith("ndd"):
        return "NDD"
    if name.startswith("dd"):
        return "DD"
    return None


def _git_state(directory: Path) -> dict[str, Any]:
    """Commit e estado do repositório git que contém `directory` (se houver)."""
    def run(*args: str) -> str:
        return subprocess.run(["git", "-C", str(directory), *args], capture_output=True,
                              text=True, check=True).stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_files": [l for l in run("status", "--porcelain", "--", ".").splitlines() if l],
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "branch": None, "dirty_files": []}


def import_legacy(cfg: CorpusConfig, *, force: bool = False) -> Path:
    """Copia o corpus inicial para `sources/legacy/` e grava o manifesto de proveniência.

    Recusa sobrescrever uma importação existente sem `force`, porque é a entrada
    congelada do experimento.
    """
    origin_dir = cfg.paths.legacy_dataset
    target = cfg.paths.sources / "legacy"
    if origin_dir is None:
        raise ValueError("defina [paths] legacy_dataset na configuração para importar o corpus inicial")
    if not origin_dir.is_dir():
        raise FileNotFoundError(f"corpus inicial não encontrado em {origin_dir}")
    if target.exists() and not force:
        raise FileExistsError(f"{target} já existe; use --force para reimportar")
    if target.exists():
        shutil.rmtree(target)

    git = _git_state(origin_dir)
    if git["dirty_files"]:
        log.warn("o corpus de origem tem alterações não commitadas: %s", git["dirty_files"])
    files: list[dict[str, Any]] = []
    for provider_dir in sorted(p for p in origin_dir.iterdir() if p.is_dir()):
        for label_file in ("dd.jsonl", "ndd.jsonl"):
            src = provider_dir / label_file
            if not src.exists():
                continue
            dst = target / provider_dir.name / label_file
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            with dst.open(encoding="utf-8") as handle:
                line_count = sum(1 for _ in handle)
            files.append({
                "provider": provider_dir.name,
                "file": f"{provider_dir.name}/{label_file}",
                "lines": line_count,
                "sha256": file_sha256(dst),
                "model": LEGACY_MODELS.get(provider_dir.name, "desconhecido"),
            })
    manifest = {
        "imported_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "origin": str(origin_dir),
        "git": git,
        "files": files,
    }
    write_json(target / "MANIFEST.json", manifest)
    log.info("corpus inicial importado: %d arquivos, commit %s", len(files), git["commit"])
    return target


def iter_source_files(cfg: CorpusConfig) -> Iterator[tuple[str, str, Label, Path]]:
    """Arquivos de fonte em ordem determinística: (origem, provedor, classe, caminho)."""
    for origin in ORIGINS:
        if origin not in cfg.sources.include_origins:
            continue
        origin_dir = cfg.paths.sources / origin
        if not origin_dir.is_dir():
            continue
        for provider_dir in sorted(p for p in origin_dir.iterdir() if p.is_dir()):
            for label in LABELS:
                for path in sorted(provider_dir.glob("*.jsonl")):
                    if _label_of_file(path) == label:
                        yield origin, provider_dir.name, label, path


def load_sources(cfg: CorpusConfig) -> tuple[list[SourceRow], list[dict[str, Any]], dict[str, int]]:
    """Lê todas as fontes.

    Returns:
        (linhas, manifesto por arquivo, contagem de linhas rejeitadas por motivo de leitura)
    """
    legacy_manifest_path = cfg.paths.sources / "legacy" / "MANIFEST.json"
    legacy_models = {}
    if legacy_manifest_path.exists():
        legacy_models = {f["provider"]: f["model"] for f in read_json(legacy_manifest_path)["files"]}

    rows: list[SourceRow] = []
    manifest: list[dict[str, Any]] = []
    rejected = {"rotulo_divergente": 0, "texto_ausente": 0}
    for origin, provider, label, path in iter_source_files(cfg):
        rel = str(path.relative_to(cfg.paths.sources))
        before = len(rows)
        for line_no, record in enumerate(read_jsonl(path), start=1):
            text = record.get("text")
            if not isinstance(text, str) or not text.strip():
                rejected["texto_ausente"] += 1
                continue
            if record.get("label", label) != label:
                rejected["rotulo_divergente"] += 1
                continue
            extra = {k: v for k, v in record.items() if k not in {"text", "label", "strategy", "model"}}
            rows.append(SourceRow(
                text_raw=text,
                label=label,
                source=provider,
                origin=origin,
                strategy=str(record.get("strategy") or ("legado" if origin == "legacy" else "desconhecida")),
                model=str(record.get("model") or legacy_models.get(provider, "desconhecido")),
                file=rel,
                line=line_no,
                extra=extra,
            ))
        manifest.append({
            "file": rel, "origin": origin, "provider": provider, "label": label,
            "rows": len(rows) - before, "sha256": file_sha256(path),
            "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(path.stat().st_mtime)),
        })
    return rows, manifest, rejected
