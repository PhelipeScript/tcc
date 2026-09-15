"""
Descrição:
    Stage 2 — normalização textual e filtros de integridade.

    Para cada registro do snapshot, preenche `text` com a forma normalizada
    (sem acento, minúscula, sem pontuação, espaços colapsados) e preserva o
    original em `text_raw`.

    Dois critérios de descarte são aplicados sobre o texto normalizado, e o
    relatório contabiliza cada um separadamente:

    - **tamanho**: `len > 5` e `>= 2` palavras. Rede de segurança contra
      fragmentos degenerados de geração, não um corte agressivo de comprimento.

    - **nao_ascii**: sobra de script estrangeiro. Depois da normalização, todo
      português legítimo é ASCII; o que resta não-ASCII são vazamentos dos
      modelos geradores — chinês, cirílico, árabe, coreano — dentro de frases
      que no resto estão em português. São defeitos de geração, e mantê-los
      injetaria dezenas de caracteres fora do vocabulário dos tokenizadores em
      português. O filtro pode ser desligado com `--keep-non-ascii`.

Entrada:
    data/01_snapshot/<provedor>/{dd,ndd}.jsonl

Saída:
    data/02_normalized/<provedor>/{dd,ndd}.jsonl
    data/02_normalized/dropped.jsonl
    data/reports/stages/stage2.json

Uso:
    python -m src.data_prep.stage2_normalize
    python -m src.data_prep.stage2_normalize --keep-non-ascii
"""

from __future__ import annotations

import argparse
from collections import Counter

from src.data_prep import console, report
from src.data_prep.config import (
    LABELS,
    PROVIDERS,
    STAGE1_SNAPSHOT_DIR,
    STAGE2_NORMALIZED_DIR,
    provider_class_path,
)
from src.data_prep.jsonl_io import Record, read_jsonl, write_dicts_jsonl, write_jsonl
from src.data_prep.normalizer import describe_filter, normalize_text, rejection_reason

#: Registros descartados, com o motivo, para auditoria.
DROPPED_PATH = STAGE2_NORMALIZED_DIR / "dropped.jsonl"

#: Motivos de descarte contabilizados, em ordem de exibição.
DROP_REASONS: tuple[str, ...] = ("tamanho", "nao_ascii")


def normalize_file(
    provider: str, label: str, *, require_ascii: bool
) -> tuple[int, Counter[str], list[dict[str, str]]]:
    """
    Descrição:
        Normaliza e filtra um arquivo de classe de um provedor.

    Args:
        provider: Nome do provedor.
        label: Rótulo da classe, `"DD"` ou `"NDD"`.
        require_ascii: Se `True`, descarta texto com script estrangeiro residual.

    Retorna:
        Tupla `(mantidos, descartados_por_motivo, registros_descartados)`, em
        que o terceiro item traz os registros rejeitados anotados com o motivo,
        para gravação no log de auditoria.
    """
    source = provider_class_path(STAGE1_SNAPSHOT_DIR, provider, label)
    destination = provider_class_path(STAGE2_NORMALIZED_DIR, provider, label)

    kept: list[Record] = []
    dropped: Counter[str] = Counter()
    dropped_rows: list[dict[str, str]] = []

    for record in read_jsonl(source):
        normalized = normalize_text(record.text_raw)
        reason = rejection_reason(normalized, require_ascii=require_ascii)
        if reason is not None:
            dropped[reason] += 1
            dropped_rows.append(
                {
                    "reason": reason,
                    "source": provider,
                    "label": label,
                    "text": normalized,
                    "text_raw": record.text_raw,
                }
            )
            continue
        kept.append(record.replace_text(normalized))

    write_jsonl(kept, destination)
    return len(kept), dropped, dropped_rows


def run(require_ascii: bool = True) -> report.StageStats:
    """
    Descrição:
        Executa o Stage 2 sobre todos os provedores e classes, montando as
        estatísticas de mantidos e de descartados por motivo.

    Args:
        require_ascii: Se `True`, descarta registros cujo texto normalizado
            ainda contenha caracteres não-ASCII.

    Retorna:
        Estatísticas do estágio, já persistidas em
        `data/reports/stages/stage2.json`.
    """
    console.header("Stage 2 — normalização e filtros de integridade")
    console.detail(f"filtro de tamanho (texto normalizado): {describe_filter()}")
    console.detail(
        "filtro de script estrangeiro: "
        + ("ativo (descarta não-ASCII)" if require_ascii else "DESLIGADO (--keep-non-ascii)")
    )

    rows: list[list[object]] = []
    totals = {label: 0 for label in LABELS}
    dropped_totals: Counter[str] = Counter()
    all_dropped: list[dict[str, str]] = []

    for provider in PROVIDERS:
        row: list[object] = [provider]
        for label in LABELS:
            kept, dropped, dropped_rows = normalize_file(
                provider, label, require_ascii=require_ascii
            )
            totals[label] += kept
            dropped_totals.update(dropped)
            all_dropped.extend(dropped_rows)
            row += [kept, dropped["tamanho"], dropped["nao_ascii"]]
        rows.append(row)

    write_dicts_jsonl(all_dropped, DROPPED_PATH)

    headers = ["provedor"]
    for label in LABELS:
        headers += [f"{label} mantidos", f"{label} -tam", f"{label} -ascii"]

    stats = report.StageStats(stage=2, name="normalizacao", totals=totals)
    stats.extra = {
        "dropped_by_reason": dict(dropped_totals),
        "dropped_total": sum(dropped_totals.values()),
        "length_filter": describe_filter(),
        "require_ascii": require_ascii,
    }
    stats.add_table("Normalização por provedor", headers, rows)
    stats.add_table(
        "Descartes por motivo",
        ["motivo", "registros"],
        [[reason, dropped_totals[reason]] for reason in DROP_REASONS],
    )
    stats.note(f"filtro de tamanho: {describe_filter()} (sobre o texto normalizado)")
    stats.note(
        f"descartados: {sum(dropped_totals.values())} registros — "
        + ", ".join(f"{reason}={dropped_totals[reason]}" for reason in DROP_REASONS)
    )
    stats.note(
        "normalização usa NFKD, que dobra tipografia de compatibilidade "
        "(`3º` -> `3o`, `m³` -> `m3`) além dos acentos canônicos"
    )
    if dropped_totals["nao_ascii"]:
        exemplos = [
            r["text_raw"] for r in all_dropped if r["reason"] == "nao_ascii"
        ][:3]
        stats.note(
            "descartes por script estrangeiro são vazamentos dos modelos geradores, ex.: "
            + "; ".join(f"`{e}`" for e in exemplos)
        )
    stats.note(f"log completo dos descartes em `{DROPPED_PATH.name}`")
    stats.render_console()
    report.save_stage_stats(stats)
    return stats


def main(argv: list[str] | None = None) -> int:
    """
    Descrição:
        Ponto de entrada de linha de comando do Stage 2.

    Args:
        argv: Argumentos de linha de comando. `None` usa `sys.argv`.

    Retorna:
        Código de saída do processo (0 em sucesso).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--keep-non-ascii",
        action="store_true",
        help="mantém registros com script estrangeiro residual (padrão: descarta)",
    )
    args = parser.parse_args(argv)
    run(require_ascii=not args.keep_non_ascii)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
