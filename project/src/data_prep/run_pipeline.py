"""
Descrição:
    Orquestrador do pipeline de preparação do dataset DDSD.

    Executa os 7 estágios em sequência e, ao final, consolida o relatório.
    Cada estágio é independente e reexecutável, então `--from-stage` e
    `--to-stage` permitem retomar a partir do meio sem refazer o que já foi
    processado.

    Sequência completa:
        1. snapshot dos 9 provedores (origem somente leitura) + manifest
        2. normalização (sem acento, minúscula, sem pontuação) + filtro de tamanho
        3. dedup exato dentro de cada arquivo
        4. merge dos provedores em 2 arquivos, um por classe
        5. dedup global por classe + remoção de conflitos DD/NDD
        6. balanceamento 1:1 por undersample proporcional
        7. split estratificado 80/10/10 por classe e provedor

Uso:
    python -m src.data_prep.run_pipeline
    python -m src.data_prep.run_pipeline --from-stage 5
    python -m src.data_prep.run_pipeline --force --seed 7

Saída:
    data/01_snapshot/ ... data/07_splits/
    data/reports/pipeline_report.{json,md}
"""

from __future__ import annotations

import argparse
import shutil
import time
from typing import Callable

from src.data_prep import console, report
from src.data_prep import (
    stage1_snapshot,
    stage2_normalize,
    stage3_dedup_per_file,
    stage4_merge,
    stage5_dedup_final,
    stage6_balance,
    stage7_split,
)
from src.data_prep.config import (
    DATA_DIR,
    SEED,
    STAGE1_SNAPSHOT_DIR,
    STAGE2_NORMALIZED_DIR,
    STAGE3_DEDUP_FILE_DIR,
    STAGE4_MERGED_DIR,
    STAGE5_DEDUP_FINAL_DIR,
    STAGE6_BALANCED_DIR,
    STAGE7_SPLITS_DIR,
)

#: Estágios do pipeline: (número, nome, função de execução, diretório de saída).
#:
#: A função recebe `seed` por palavra-chave apenas nos estágios que dependem de
#: sorteio (6 e 7); os demais ignoram o argumento.
STAGES: tuple[tuple[int, str, Callable[..., report.StageStats], object], ...] = (
    (1, "snapshot", stage1_snapshot.run, STAGE1_SNAPSHOT_DIR),
    (2, "normalizacao", stage2_normalize.run, STAGE2_NORMALIZED_DIR),
    (3, "dedup_por_arquivo", stage3_dedup_per_file.run, STAGE3_DEDUP_FILE_DIR),
    (4, "merge", stage4_merge.run, STAGE4_MERGED_DIR),
    (5, "dedup_global_e_conflitos", stage5_dedup_final.run, STAGE5_DEDUP_FINAL_DIR),
    (6, "balanceamento", stage6_balance.run, STAGE6_BALANCED_DIR),
    (7, "split", stage7_split.run, STAGE7_SPLITS_DIR),
)

#: Estágios cujo resultado depende da semente aleatória.
_SEEDED_STAGES: frozenset[int] = frozenset({6, 7})

#: Estágio que aplica o filtro de script estrangeiro.
_ASCII_STAGE: int = 2


def clean_stage_outputs(from_stage: int, to_stage: int) -> None:
    """
    Descrição:
        Remove os diretórios de saída dos estágios no intervalo indicado, para
        que a rodada comece de um estado limpo.

        Só apaga dentro de `data/`, nunca nas pastas de origem dos provedores.

    Args:
        from_stage: Primeiro estágio a limpar (inclusive).
        to_stage: Último estágio a limpar (inclusive).
    """
    for number, name, _run, directory in STAGES:
        if from_stage <= number <= to_stage and directory.exists():
            shutil.rmtree(directory)
            console.detail(f"removido {directory.relative_to(DATA_DIR.parent)} (stage {number} {name})")


def run_pipeline(
    from_stage: int = 1,
    to_stage: int = 7,
    seed: int = SEED,
    force: bool = False,
    require_ascii: bool = True,
) -> list[report.StageStats]:
    """
    Descrição:
        Executa o pipeline no intervalo de estágios indicado e grava o relatório
        consolidado.

    Args:
        from_stage: Primeiro estágio a executar (1 a 7).
        to_stage: Último estágio a executar (1 a 7).
        seed: Semente usada nos estágios 6 e 7.
        force: Se `True`, apaga os diretórios de saída do intervalo antes de
            começar.
        require_ascii: Repassado ao Stage 2; se `False`, mantém registros com
            script estrangeiro residual.

    Retorna:
        Lista com as estatísticas de cada estágio executado.

    Levanta:
        ValueError: Se o intervalo de estágios for inválido.
    """
    if not 1 <= from_stage <= to_stage <= len(STAGES):
        raise ValueError(
            f"intervalo inválido: --from-stage {from_stage} --to-stage {to_stage} "
            f"(esperado 1 <= from <= to <= {len(STAGES)})"
        )

    console.header("Pipeline de preparação do dataset DDSD")
    console.info(f"estágios {from_stage} a {to_stage}  semente {seed}")
    console.info(f"saída em {DATA_DIR}")

    if force:
        console.header("Limpando saídas anteriores (--force)")
        clean_stage_outputs(from_stage, to_stage)

    collected: list[report.StageStats] = []
    started = time.perf_counter()

    for number, _name, run_stage, _directory in STAGES:
        if not from_stage <= number <= to_stage:
            continue
        kwargs: dict[str, object] = {}
        if number in _SEEDED_STAGES:
            kwargs["seed"] = seed
        if number == _ASCII_STAGE:
            kwargs["require_ascii"] = require_ascii
        collected.append(run_stage(**kwargs))

    elapsed = time.perf_counter() - started

    console.header("Relatório consolidado")
    json_path, md_path = report.write_report()
    console.ok(f"{json_path}")
    console.ok(f"{md_path}")
    console.ok(f"pipeline concluído em {elapsed:.1f}s")

    return collected


def main(argv: list[str] | None = None) -> int:
    """
    Descrição:
        Ponto de entrada de linha de comando do orquestrador.

    Args:
        argv: Argumentos de linha de comando. `None` usa `sys.argv`.

    Retorna:
        Código de saída do processo (0 em sucesso).
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--from-stage", type=int, default=1, choices=range(1, len(STAGES) + 1),
        metavar="N", help="primeiro estágio a executar (padrão: 1)",
    )
    parser.add_argument(
        "--to-stage", type=int, default=len(STAGES), choices=range(1, len(STAGES) + 1),
        metavar="N", help=f"último estágio a executar (padrão: {len(STAGES)})",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help=f"semente dos estágios 6 e 7 (padrão: {SEED})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="apaga as saídas dos estágios do intervalo antes de executar",
    )
    parser.add_argument(
        "--keep-non-ascii", action="store_true",
        help="Stage 2: mantém registros com script estrangeiro residual",
    )
    args = parser.parse_args(argv)

    run_pipeline(
        from_stage=args.from_stage,
        to_stage=args.to_stage,
        seed=args.seed,
        force=args.force,
        require_ascii=not args.keep_non_ascii,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
