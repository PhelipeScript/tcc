"""
Descrição:
    Stage 3 — deduplicação exata dentro de cada arquivo.

    Remove textos repetidos dentro de um mesmo `dd.jsonl` ou `ndd.jsonl` de um
    provedor. A chave de comparação é o texto já normalizado pelo Stage 2, e a
    primeira ocorrência é a que permanece — como a leitura segue a ordem do
    arquivo, o resultado é determinístico.

    Este estágio existe porque os scripts de geração acrescentam ao arquivo sem
    consultar um conjunto de textos já vistos, o que produz repetições entre
    lotes e entre reexecuções (concentradas em `glm`, `groq` e `qwen`).

    A deduplicação entre provedores e entre classes é responsabilidade do
    Stage 5, após o merge.

Entrada:
    data/02_normalized/<provedor>/{dd,ndd}.jsonl

Saída:
    data/03_dedup_file/<provedor>/{dd,ndd}.jsonl
    data/reports/stages/stage3.json

Uso:
    python -m src.data_prep.stage3_dedup_per_file
"""

from __future__ import annotations

import argparse

from src.data_prep import console, report
from src.data_prep.config import (
    LABELS,
    PROVIDERS,
    STAGE2_NORMALIZED_DIR,
    STAGE3_DEDUP_FILE_DIR,
    provider_class_path,
)
from src.data_prep.jsonl_io import Record, read_jsonl, write_jsonl


def dedup_file(provider: str, label: str) -> tuple[int, int]:
    """
    Descrição:
        Deduplica um arquivo de classe de um provedor pelo texto normalizado,
        mantendo a primeira ocorrência de cada texto.

    Args:
        provider: Nome do provedor.
        label: Rótulo da classe, `"DD"` ou `"NDD"`.

    Retorna:
        Tupla `(unicos, duplicados_removidos)`.
    """
    source = provider_class_path(STAGE2_NORMALIZED_DIR, provider, label)
    destination = provider_class_path(STAGE3_DEDUP_FILE_DIR, provider, label)

    seen: set[str] = set()
    unique: list[Record] = []
    duplicates = 0

    for record in read_jsonl(source):
        if record.text in seen:
            duplicates += 1
            continue
        seen.add(record.text)
        unique.append(record)

    write_jsonl(unique, destination)
    return len(unique), duplicates


def run() -> report.StageStats:
    """
    Descrição:
        Executa o Stage 3 sobre todos os provedores e classes.

    Retorna:
        Estatísticas do estágio, já persistidas em
        `data/reports/stages/stage3.json`.
    """
    console.header("Stage 3 — dedup exato por arquivo")

    rows: list[list[object]] = []
    totals = {label: 0 for label in LABELS}
    removed_totals = {label: 0 for label in LABELS}

    for provider in PROVIDERS:
        row: list[object] = [provider]
        for label in LABELS:
            unique, duplicates = dedup_file(provider, label)
            totals[label] += unique
            removed_totals[label] += duplicates
            row += [unique, duplicates]
        rows.append(row)

    headers = ["provedor"]
    for label in LABELS:
        headers += [f"{label} únicos", f"{label} duplicados"]

    stats = report.StageStats(stage=3, name="dedup_por_arquivo", totals=totals)
    stats.extra = {"removed": removed_totals}
    stats.add_table("Deduplicação intra-arquivo", headers, rows)
    stats.note(
        "duplicados removidos: "
        + "  ".join(f"{k}={v}" for k, v in removed_totals.items())
        + f"  (total={sum(removed_totals.values())})"
    )
    stats.note("chave de dedup: texto normalizado; primeira ocorrência prevalece")
    stats.render_console()
    report.save_stage_stats(stats)
    return stats


def main(argv: list[str] | None = None) -> int:
    """
    Descrição:
        Ponto de entrada de linha de comando do Stage 3.

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
