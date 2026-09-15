"""
Descrição:
    Stage 6 — balanceamento das classes por undersample.

    Reduz a classe majoritária ao tamanho exato da minoritária, produzindo um
    dataset 1:1. O sorteio **não** é uniforme sobre o corpus inteiro: é
    proporcional por provedor, de modo que a composição de fontes da classe
    majoritária seja preservada.

    Por que proporcional: um sorteio uniforme distorceria o mix de provedores
    (provedores com mais registros perderiam proporcionalmente o mesmo que os
    menores, mas em números absolutos muito maiores), e a estratificação por
    fonte do Stage 7 passaria a operar sobre uma distribuição diferente da
    original. Com quotas proporcionais, a proporção de cada provedor dentro da
    classe majoritária é mantida.

    As quotas são calculadas pelo método do maior resto, garantindo que a soma
    seja exatamente igual ao tamanho da classe minoritária — sem sobra nem
    falta por arredondamento.

    Os registros descartados são gravados em `discarded.jsonl`, para que o
    volume e a composição do descarte possam ser reportados no TCC.

Entrada:
    data/05_dedup_final/{dd,ndd}.jsonl

Saída:
    data/06_balanced/{dd,ndd}.jsonl
    data/06_balanced/discarded.jsonl
    data/reports/stages/stage6.json

Uso:
    python -m src.data_prep.stage6_balance [--seed 42]
"""

from __future__ import annotations

import argparse
import random
from collections import Counter

from src.data_prep import console, report
from src.data_prep.config import (
    DISCARDED_PATH,
    LABELS,
    SEED,
    STAGE5_DEDUP_FINAL_DIR,
    STAGE6_BALANCED_DIR,
    merged_class_path,
)
from src.data_prep.jsonl_io import Record, read_jsonl, write_jsonl


def proportional_quotas(counts: dict[str, int], target: int) -> dict[str, int]:
    """
    Descrição:
        Distribui `target` vagas entre os provedores proporcionalmente às suas
        contagens, usando o método do maior resto.

        O método garante que a soma das quotas seja exatamente `target`: primeiro
        cada provedor recebe a parte inteira da sua fração ideal, e as vagas
        restantes vão para os provedores com maior parte fracionária (com
        desempate pelo nome do provedor, para ser determinístico).

    Args:
        counts: Contagem de registros disponíveis por provedor.
        target: Número total de registros a manter.

    Retorna:
        Quota por provedor, somando exatamente `target`. Nenhuma quota excede a
        contagem disponível daquele provedor.

    Levanta:
        ValueError: Se `target` for maior que o total disponível.
    """
    total = sum(counts.values())
    if target > total:
        raise ValueError(f"alvo {target} maior que o total disponível {total}")

    exact = {p: target * n / total for p, n in counts.items()}
    quotas = {p: int(value) for p, value in exact.items()}

    remaining = target - sum(quotas.values())
    # Ordena por parte fracionária decrescente; o nome do provedor desempata.
    by_remainder = sorted(
        counts, key=lambda p: (-(exact[p] - quotas[p]), p)
    )
    for provider in by_remainder:
        if remaining <= 0:
            break
        if quotas[provider] < counts[provider]:
            quotas[provider] += 1
            remaining -= 1

    return quotas


def undersample(
    records: list[Record], target: int, seed: int
) -> tuple[list[Record], list[Record], dict[str, int]]:
    """
    Descrição:
        Sorteia `target` registros da classe majoritária, com quotas
        proporcionais por provedor.

        A ordem original dos registros mantidos é preservada (o resultado é
        ordenado pelo índice de origem), de modo que o arquivo de saída seja
        reproduzível byte a byte para a mesma semente.

    Args:
        records: Registros da classe majoritária, na ordem do Stage 5.
        target: Quantidade a manter.
        seed: Semente do sorteio.

    Retorna:
        Tupla `(mantidos, descartados, quotas_por_provedor)`.
    """
    by_provider: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        by_provider.setdefault(record.source, []).append(index)

    counts = {provider: len(indices) for provider, indices in by_provider.items()}
    quotas = proportional_quotas(counts, target)

    rng = random.Random(seed)
    kept_indices: set[int] = set()
    for provider in sorted(by_provider):
        indices = by_provider[provider]
        kept_indices.update(rng.sample(indices, quotas[provider]))

    kept = [records[i] for i in range(len(records)) if i in kept_indices]
    discarded = [records[i] for i in range(len(records)) if i not in kept_indices]
    return kept, discarded, quotas


def run(seed: int = SEED) -> report.StageStats:
    """
    Descrição:
        Executa o Stage 6: identifica a classe majoritária, aplica o undersample
        proporcional e grava as duas classes balanceadas.

    Args:
        seed: Semente do sorteio, para reprodutibilidade.

    Retorna:
        Estatísticas do estágio, já persistidas em
        `data/reports/stages/stage6.json`.
    """
    console.header("Stage 6 — balanceamento 1:1 por undersample")

    loaded: dict[str, list[Record]] = {
        label: list(read_jsonl(merged_class_path(STAGE5_DEDUP_FINAL_DIR, label)))
        for label in LABELS
    }
    sizes = {label: len(records) for label, records in loaded.items()}

    majority = max(sizes, key=lambda label: sizes[label])
    minority = min(sizes, key=lambda label: sizes[label])
    target = sizes[minority]

    console.detail(
        f"majoritária={majority} ({sizes[majority]})  minoritária={minority} ({sizes[minority]})"
    )

    kept, discarded, quotas = undersample(loaded[majority], target, seed)

    totals: dict[str, int] = {}
    totals[majority] = write_jsonl(kept, merged_class_path(STAGE6_BALANCED_DIR, majority))
    totals[minority] = write_jsonl(
        loaded[minority], merged_class_path(STAGE6_BALANCED_DIR, minority)
    )
    write_jsonl(discarded, DISCARDED_PATH)

    before = Counter(r.source for r in loaded[majority])
    after = Counter(r.source for r in kept)
    minority_mix = Counter(r.source for r in loaded[minority])

    stats = report.StageStats(stage=6, name="balanceamento", totals=totals)
    stats.extra = {
        "majority_class": majority,
        "minority_class": minority,
        "seed": seed,
        "discarded": len(discarded),
        "quotas": quotas,
    }
    stats.add_table(
        f"Undersample da classe {majority} — composição por provedor",
        ["provedor", "antes", "quota", "depois", "% antes", "% depois"],
        [
            [
                provider,
                before[provider],
                quotas.get(provider, 0),
                after[provider],
                f"{100 * before[provider] / sizes[majority]:.2f}",
                f"{100 * after[provider] / max(target, 1):.2f}",
            ]
            for provider in sorted(before)
        ],
    )
    stats.add_table(
        "Composição final por provedor e classe",
        ["provedor", *LABELS, "total"],
        [
            [
                provider,
                *[
                    (after[provider] if label == majority else minority_mix[provider])
                    for label in LABELS
                ],
                after[provider] + minority_mix[provider],
            ]
            for provider in sorted(set(before) | set(minority_mix))
        ],
    )
    stats.note(
        f"undersample de {majority}: {sizes[majority]} -> {target} "
        f"({len(discarded)} registros descartados, semente {seed})"
    )
    stats.note(
        "quotas proporcionais por provedor (método do maior resto), preservando o mix de fontes"
    )
    stats.note(f"registros descartados preservados em `{DISCARDED_PATH.name}` para auditoria")
    stats.render_console()
    report.save_stage_stats(stats)

    if totals[majority] != totals[minority]:
        raise AssertionError(
            f"balanceamento falhou: {majority}={totals[majority]} != {minority}={totals[minority]}"
        )

    return stats


def main(argv: list[str] | None = None) -> int:
    """
    Descrição:
        Ponto de entrada de linha de comando do Stage 6.

    Args:
        argv: Argumentos de linha de comando. `None` usa `sys.argv`.

    Retorna:
        Código de saída do processo (0 em sucesso).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--seed", type=int, default=SEED, help=f"semente do sorteio (padrão: {SEED})"
    )
    args = parser.parse_args(argv)
    run(seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
