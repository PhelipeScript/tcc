"""
Descrição:
    Stage 5 — deduplicação global e remoção de conflitos de rótulo.

    Executa as duas sub-etapas definidas para o dataset, nesta ordem:

    5a. **Dedup por classe** — nenhum texto pode aparecer duas vezes na mesma
        classe. Como o Stage 3 já eliminou repetições dentro de cada arquivo, o
        que este passo remove é a sobreposição *entre provedores* (por exemplo,
        `openrouter` e `qwen` gerando o mesmo comando). Essa sobreposição é
        concentrada na classe DD, cujo espaço de comandos imperativos é pequeno.

    5b. **Remoção de conflito** — nenhum texto pode aparecer nas duas classes.
        Um texto rotulado como DD por um provedor e como NDD por outro é
        genuinamente ambíguo (a classe `AMB` da taxonomia do projeto) e é
        removido de **ambas** as classes, nunca atribuído a uma delas. Os casos
        são registrados em `conflicts.jsonl`, com os provedores de cada lado —
        material qualitativo direto para a discussão sobre a fronteira DD/NDD.

    Rodar o dedup global aqui, antes do split, é o que impede vazamento entre
    treino e teste: um texto duplicado que caísse em partições diferentes
    inflaria artificialmente as métricas.

    Critério de desempate: primeira ocorrência na ordem determinística do
    Stage 4 (`config.PROVIDERS` ordenado). O campo `source` do registro que
    permanece é, portanto, o do provedor alfabeticamente anterior entre os que
    produziram aquele texto.

Entrada:
    data/04_merged/{dd,ndd}.jsonl

Saída:
    data/05_dedup_final/{dd,ndd}.jsonl
    data/05_dedup_final/conflicts.jsonl
    data/reports/stages/stage5.json

Uso:
    python -m src.data_prep.stage5_dedup_final
"""

from __future__ import annotations

import argparse

from src.data_prep import console, report
from src.data_prep.config import (
    CONFLICTS_PATH,
    LABELS,
    STAGE4_MERGED_DIR,
    STAGE5_DEDUP_FINAL_DIR,
    merged_class_path,
)
from src.data_prep.jsonl_io import Record, read_jsonl, write_dicts_jsonl, write_jsonl


def dedup_within_class(label: str) -> tuple[dict[str, Record], dict[str, list[str]], int]:
    """
    Descrição:
        Sub-etapa 5a: deduplica uma classe inteira pelo texto normalizado,
        mantendo a primeira ocorrência.

    Args:
        label: Rótulo da classe, `"DD"` ou `"NDD"`.

    Retorna:
        Tupla `(unicos, provedores_por_texto, duplicados_removidos)`, em que
        `unicos` mapeia texto -> registro mantido (preservando a ordem de
        inserção) e `provedores_por_texto` lista todos os provedores que
        produziram cada texto, mesmo os descartados.
    """
    source = merged_class_path(STAGE4_MERGED_DIR, label)

    unique: dict[str, Record] = {}
    providers: dict[str, list[str]] = {}
    duplicates = 0

    for record in read_jsonl(source):
        if record.text in unique:
            duplicates += 1
        else:
            unique[record.text] = record
            providers[record.text] = []
        if record.source not in providers[record.text]:
            providers[record.text].append(record.source)

    return unique, providers, duplicates


def find_conflicts(
    per_class: dict[str, dict[str, Record]],
) -> list[str]:
    """
    Descrição:
        Sub-etapa 5b (detecção): encontra os textos presentes nas duas classes.

    Args:
        per_class: Mapeamento `label -> {texto: registro}` produzido por
            `dedup_within_class`.

    Retorna:
        Lista ordenada dos textos em conflito.
    """
    dd_texts = set(per_class["DD"])
    ndd_texts = set(per_class["NDD"])
    return sorted(dd_texts & ndd_texts)


def run() -> report.StageStats:
    """
    Descrição:
        Executa o Stage 5 completo: dedup por classe, detecção e remoção de
        conflitos, e gravação dos artefatos.

    Retorna:
        Estatísticas do estágio, já persistidas em
        `data/reports/stages/stage5.json`.
    """
    console.header("Stage 5 — dedup global por classe e remoção de conflitos")

    per_class: dict[str, dict[str, Record]] = {}
    providers_per_class: dict[str, dict[str, list[str]]] = {}
    duplicates: dict[str, int] = {}

    for label in LABELS:
        unique, providers, removed = dedup_within_class(label)
        per_class[label] = unique
        providers_per_class[label] = providers
        duplicates[label] = removed
        console.detail(
            f"{label}: {len(unique)} únicos, {removed} duplicados entre provedores removidos"
        )

    conflicts = find_conflicts(per_class)

    # Grava o log de conflitos antes de remover, para que os casos fiquem
    # auditáveis com a origem de cada lado.
    conflict_rows = [
        {
            "text": text,
            "dd_sources": providers_per_class["DD"][text],
            "ndd_sources": providers_per_class["NDD"][text],
            "text_raw_dd": per_class["DD"][text].text_raw,
            "text_raw_ndd": per_class["NDD"][text].text_raw,
        }
        for text in conflicts
    ]
    write_dicts_jsonl(conflict_rows, CONFLICTS_PATH)

    conflict_set = set(conflicts)
    totals: dict[str, int] = {}
    for label in LABELS:
        kept = [rec for text, rec in per_class[label].items() if text not in conflict_set]
        totals[label] = write_jsonl(kept, merged_class_path(STAGE5_DEDUP_FINAL_DIR, label))

    ratio = (
        max(totals.values()) / min(totals.values()) if min(totals.values()) > 0 else float("inf")
    )

    stats = report.StageStats(stage=5, name="dedup_global_e_conflitos", totals=totals)
    stats.extra = {
        "cross_provider_duplicates": duplicates,
        "conflicts": len(conflicts),
        "records_dropped_by_conflict": len(conflicts) * 2,
        "imbalance_ratio": round(ratio, 4),
    }
    stats.add_table(
        "Sub-etapa 5a — dedup entre provedores, por classe",
        ["classe", "únicos", "duplicados removidos"],
        [[label, len(per_class[label]), duplicates[label]] for label in LABELS],
    )
    stats.add_table(
        "Sub-etapa 5b — conflitos de rótulo (removidos das duas classes)",
        ["texto", "DD gerado por", "NDD gerado por"],
        [
            [row["text"], ", ".join(row["dd_sources"]), ", ".join(row["ndd_sources"])]
            for row in conflict_rows
        ],
    )
    stats.note(
        f"{len(conflicts)} textos apareceram nas duas classes e foram removidos de ambas "
        f"({len(conflicts) * 2} registros); log em `{CONFLICTS_PATH.name}`"
    )
    stats.note(f"desbalanceamento após o dedup global: {ratio:.3f}:1")
    stats.note(
        "desempate do dedup: primeira ocorrência na ordem alfabética de provedores, "
        "portanto o campo `source` mantido é o do provedor anterior"
    )
    stats.render_console()
    report.save_stage_stats(stats)
    return stats


def main(argv: list[str] | None = None) -> int:
    """
    Descrição:
        Ponto de entrada de linha de comando do Stage 5.

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
