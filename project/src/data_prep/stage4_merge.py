"""
Descrição:
    Stage 4 — merge dos provedores em um único corpus por classe.

    Concatena os arquivos já normalizados e deduplicados dos 9 provedores em
    apenas dois arquivos: `dd.jsonl` e `ndd.jsonl`. O campo `source` de cada
    registro preserva a origem, que continua sendo necessária para o undersample
    proporcional (Stage 6) e para a estratificação do split (Stage 7).

    A ordem de leitura segue `config.PROVIDERS`, que é fixa — logo o arquivo
    resultante é byte-a-byte reproduzível para a mesma entrada.

    Este estágio não remove nada: duplicatas entre provedores e conflitos de
    rótulo entre classes são tratados no Stage 5.

Entrada:
    data/03_dedup_file/<provedor>/{dd,ndd}.jsonl

Saída:
    data/04_merged/{dd,ndd}.jsonl
    data/reports/stages/stage4.json

Uso:
    python -m src.data_prep.stage4_merge
"""

from __future__ import annotations

import argparse
from typing import Iterator

from src.data_prep import console, report
from src.data_prep.config import (
    LABELS,
    PROVIDERS,
    STAGE3_DEDUP_FILE_DIR,
    STAGE4_MERGED_DIR,
    merged_class_path,
    provider_class_path,
)
from src.data_prep.jsonl_io import Record, read_jsonl, write_jsonl


def iter_class_records(label: str) -> Iterator[Record]:
    """
    Descrição:
        Percorre os registros de uma classe em todos os provedores, na ordem
        fixa de `config.PROVIDERS`.

    Args:
        label: Rótulo da classe, `"DD"` ou `"NDD"`.

    Produz:
        Registros de todos os provedores, em sequência.
    """
    for provider in PROVIDERS:
        source = provider_class_path(STAGE3_DEDUP_FILE_DIR, provider, label)
        yield from read_jsonl(source)


def merge_class(label: str) -> dict[str, int]:
    """
    Descrição:
        Concatena os arquivos de uma classe em um único JSONL consolidado.

    Args:
        label: Rótulo da classe, `"DD"` ou `"NDD"`.

    Retorna:
        Contagem de registros por provedor, mais a chave `"total"`.
    """
    destination = merged_class_path(STAGE4_MERGED_DIR, label)
    per_provider: dict[str, int] = {}
    records: list[Record] = []

    for provider in PROVIDERS:
        source = provider_class_path(STAGE3_DEDUP_FILE_DIR, provider, label)
        before = len(records)
        records.extend(read_jsonl(source))
        per_provider[provider] = len(records) - before

    per_provider["total"] = write_jsonl(records, destination)
    return per_provider


def run() -> report.StageStats:
    """
    Descrição:
        Executa o Stage 4 para as duas classes.

    Retorna:
        Estatísticas do estágio, já persistidas em
        `data/reports/stages/stage4.json`.
    """
    console.header("Stage 4 — merge dos provedores")

    per_label: dict[str, dict[str, int]] = {}
    for label in LABELS:
        per_label[label] = merge_class(label)

    totals = {label: per_label[label]["total"] for label in LABELS}

    stats = report.StageStats(stage=4, name="merge", totals=totals)
    stats.extra = {"per_provider": per_label}
    stats.add_table(
        "Composição do corpus consolidado por provedor",
        ["provedor", *LABELS, "total"],
        [
            [
                provider,
                *[per_label[label][provider] for label in LABELS],
                sum(per_label[label][provider] for label in LABELS),
            ]
            for provider in PROVIDERS
        ],
    )
    stats.note(
        f"consolidado em 2 arquivos: `{merged_class_path(STAGE4_MERGED_DIR, 'DD').name}` "
        f"e `{merged_class_path(STAGE4_MERGED_DIR, 'NDD').name}`"
    )
    stats.note("nenhum registro removido neste estágio (apenas concatenação)")
    stats.render_console()
    report.save_stage_stats(stats)
    return stats


def main(argv: list[str] | None = None) -> int:
    """
    Descrição:
        Ponto de entrada de linha de comando do Stage 4.

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
