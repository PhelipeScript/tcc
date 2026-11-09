"""Congelamento do corpus: cópia imutável do build e das fontes que o produziram.

O build registra o SHA-256 de cada arquivo de fonte. Como a geração grava nas fontes em modo
append, um arquivo pode mudar entre o build e o congelamento. Por isso, antes de copiar, o
congelamento confere cada SHA-256 e recusa fontes que mudaram. Depois arquiva essas fontes em
`sources.tar.gz`, e o corpus congelado passa a ser reconstruível sem depender de `data/corpus/sources`.
"""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path

from ddsd.common.io import file_sha256, read_json

SOURCES_ARCHIVE = "sources.tar.gz"


class SourcesChanged(RuntimeError):
    """Uma fonte listada no manifesto do build mudou ou sumiu depois do build."""


def changed_sources(manifest: dict, sources_root: Path) -> list[str]:
    """Arquivos de fonte cujo conteúdo atual não bate com o SHA-256 gravado no build."""
    changed = []
    for entry in manifest["sources"]:
        path = sources_root / entry["file"]
        if not path.is_file() or file_sha256(path) != entry["sha256"]:
            changed.append(entry["file"])
    return changed


def freeze_corpus(build: Path, frozen: Path, sources_root: Path, *, force: bool = False,
                  reports_dir: Path | None = None) -> Path:
    """Copia `build` para `frozen` e arquiva as fontes exatas do build. Devolve o caminho do arquivo.

    Com `reports_dir`, copia também `report.md` e `manifest.json` para lá (o diretório `data/` não é
    versionado; `reports/` é).
    """
    manifest_path = build / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"nenhum build em {build}; rode `ddsd corpus build`")
    if frozen.exists() and not force:
        raise FileExistsError(f"{frozen} já existe; use --force para substituir o corpus congelado")
    manifest = read_json(manifest_path)
    changed = changed_sources(manifest, sources_root)
    if changed:
        raise SourcesChanged(f"{len(changed)} fonte(s) mudaram depois do build (ex.: {changed[:3]}); "
                             "rode `ddsd corpus build` de novo antes de congelar")
    staging = frozen.with_name(frozen.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(build, staging)
    archive = staging / SOURCES_ARCHIVE
    with tarfile.open(archive, "w:gz") as tar:
        for entry in manifest["sources"]:
            tar.add(sources_root / entry["file"], arcname=entry["file"])
        legacy_manifest = sources_root / "legacy" / "MANIFEST.json"
        if legacy_manifest.is_file():
            tar.add(legacy_manifest, arcname="legacy/MANIFEST.json")
    if changed_sources(manifest, sources_root):  # uma escrita concorrente durante a cópia
        shutil.rmtree(staging)
        raise SourcesChanged("fontes mudaram durante o congelamento; rode o build de novo")
    if frozen.exists():
        shutil.rmtree(frozen)
    staging.rename(frozen)
    if reports_dir is not None:
        export_reports(frozen, reports_dir)
    return frozen / SOURCES_ARCHIVE


def export_reports(frozen: Path, reports_dir: Path) -> None:
    """Copia o relatório e o manifesto do corpus congelado para o diretório de resultados."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    for name in ("report.md", "manifest.json"):
        shutil.copy2(frozen / name, reports_dir / name)
