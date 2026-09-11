"""
Descrição:
    Stage 1 — snapshot dos corpora de origem.

    Copia os arquivos `dd.jsonl` e `ndd.jsonl` dos 9 provedores para
    `data/01_snapshot/`, convertendo o formato legado de duas chaves
    (`{"text", "label"}`) no registro canônico (`{"text", "text_raw", "label",
    "source"}`).

    As pastas de origem (`project/src/dataset/glm/`, `project/src/dataset/groq/`, ...) são tratadas como
    SOMENTE LEITURA — este é o único módulo do pipeline que as abre, e apenas
    para leitura.

    Por que um snapshot com manifest: os scripts `src/dataset/*_generate.py`
    podem estar rodando e acrescentando linhas aos arquivos de origem durante a
    execução. O manifest registra contagem, tamanho, mtime e sha256 de cada
    arquivo no instante da cópia, tornando a rodada auditável e reproduzível.
    Para incorporar dados gerados depois, basta reexecutar o pipeline.

Entrada:
    project/<provedor>/{dd,ndd}.jsonl   (formato legado, 2 chaves)

Saída:
    data/01_snapshot/<provedor>/{dd,ndd}.jsonl
    data/01_snapshot/manifest.json
    data/reports/stages/stage1.json

Uso:
    python -m src.data_prep.stage1_snapshot
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data_prep import console, report
from src.data_prep.config import (
    CLASS_FILES,
    LABELS,
    MANIFEST_PATH,
    PROVIDERS,
    STAGE1_SNAPSHOT_DIR,
    provider_class_path,
    provider_dir,
)
from src.data_prep.jsonl_io import read_flat_jsonl, write_jsonl

#: Tamanho do bloco de leitura ao calcular o sha256 dos arquivos de origem.
_HASH_CHUNK_BYTES: int = 1024 * 1024


def file_sha256(path: Path) -> str:
    """
    Descrição:
        Calcula o SHA-256 do conteúdo de um arquivo, lendo em blocos para não
        carregá-lo inteiro na memória.

    Args:
        path: Caminho do arquivo.

    Retorna:
        Digest hexadecimal do conteúdo.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sources() -> None:
    """
    Descrição:
        Verifica que todos os arquivos de origem esperados existem antes de
        iniciar a cópia, falhando cedo e com mensagem explícita.

    Levanta:
        FileNotFoundError: Se algum diretório de provedor ou arquivo de classe
            estiver ausente.
    """
    missing: list[str] = []
    for provider in PROVIDERS:
        directory = provider_dir(provider)
        if not directory.is_dir():
            missing.append(f"{provider}/ (diretório inexistente)")
            continue
        for label in LABELS:
            path = directory / CLASS_FILES[label]
            if not path.is_file():
                missing.append(f"{provider}/{CLASS_FILES[label]}")
    if missing:
        raise FileNotFoundError(
            "arquivos de origem ausentes:\n  - " + "\n  - ".join(missing)
        )


def snapshot_file(provider: str, label: str) -> dict[str, Any]:
    """
    Descrição:
        Copia um arquivo de classe de um provedor para o diretório do Stage 1,
        promovendo cada linha ao registro canônico, e coleta os metadados desse
        arquivo para o manifest.

    Args:
        provider: Nome do provedor.
        label: Rótulo da classe, `"DD"` ou `"NDD"`.

    Retorna:
        Entrada do manifest com `provider`, `label`, `source_path`, `records`,
        `bytes`, `mtime` e `sha256`.

    Levanta:
        ValueError: Se alguma linha declarar um rótulo diferente do esperado
            para o arquivo (inconsistência entre nome do arquivo e campo
            `label`).
    """
    source = provider_dir(provider) / CLASS_FILES[label]
    destination = provider_class_path(STAGE1_SNAPSHOT_DIR, provider, label)

    stat = source.stat()
    digest = file_sha256(source)

    records = []
    for record in read_flat_jsonl(source, provider):
        if record.label != label:
            raise ValueError(
                f"{source}: rótulo {record.label!r} inesperado em arquivo de {label}"
            )
        records.append(record)

    written = write_jsonl(records, destination)

    return {
        "provider": provider,
        "label": label,
        "source_path": str(source),
        "records": written,
        "bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(
            timespec="seconds"
        ),
        "sha256": digest,
    }


def run() -> report.StageStats:
    """
    Descrição:
        Executa o Stage 1 por completo: valida as origens, copia os 18 arquivos,
        grava o manifest e monta as estatísticas do estágio.

    Retorna:
        Estatísticas do estágio, já persistidas em `data/reports/stages/stage1.json`.

    Levanta:
        FileNotFoundError: Se algum arquivo de origem estiver ausente.
        ValueError: Se um arquivo contiver rótulo inconsistente com seu nome.
    """
    console.header("Stage 1 — snapshot dos corpora de origem")
    verify_sources()

    snapshot_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entries: list[dict[str, Any]] = []
    for provider in PROVIDERS:
        for label in LABELS:
            entries.append(snapshot_file(provider, label))

    totals = {label: sum(e["records"] for e in entries if e["label"] == label) for label in LABELS}

    manifest = {
        "snapshot_at": snapshot_at,
        "providers": list(PROVIDERS),
        "totals": totals,
        "files": entries,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    stats = report.StageStats(stage=1, name="snapshot", totals=totals)
    stats.extra = {"snapshot_at": snapshot_at, "manifest": str(MANIFEST_PATH)}
    stats.add_table(
        "Registros copiados por provedor",
        ["provedor", *LABELS, "total"],
        [
            [
                provider,
                *[
                    next(
                        e["records"]
                        for e in entries
                        if e["provider"] == provider and e["label"] == label
                    )
                    for label in LABELS
                ],
                sum(e["records"] for e in entries if e["provider"] == provider),
            ]
            for provider in PROVIDERS
        ],
    )
    stats.note(f"snapshot tomado em {snapshot_at}")
    stats.note(f"manifest com contagem, mtime e sha256 por arquivo: `{MANIFEST_PATH.name}`")
    stats.render_console()
    report.save_stage_stats(stats)
    return stats


def main(argv: list[str] | None = None) -> int:
    """
    Descrição:
        Ponto de entrada de linha de comando do Stage 1.

    Args:
        argv: Argumentos de linha de comando. `None` usa `sys.argv`.

    Retorna:
        Código de saída do processo (0 em sucesso).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args(argv)
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
