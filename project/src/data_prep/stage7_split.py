"""
Descrição:
    Stage 7 — particionamento em treino, validação e teste.

    Divide o dataset balanceado em 80% / 10% / 10%, com estratificação pela
    chave composta `"<classe>|<provedor>"`. Estratificar pelas duas dimensões
    ao mesmo tempo — e não só pela classe — atende à exigência metodológica de
    que cada partição contenha amostras de todas as origens na mesma proporção,
    de modo que o conjunto de teste não fique dominado por um provedor.

    A divisão é feita em duas passadas com `sklearn.model_selection.
    train_test_split`: primeiro 80/20, depois os 20% restantes em 10/10. Ambas
    usam `random_state` fixo, portanto o resultado é reproduzível.

    Ao final, o estágio verifica que as três partições são disjuntas quanto ao
    texto normalizado. Essa verificação é a rede de segurança contra vazamento:
    o dedup global do Stage 5 deveria tê-lo eliminado, e aqui isso é comprovado
    em vez de assumido.

Entrada:
    data/06_balanced/{dd,ndd}.jsonl

Saída:
    data/07_splits/{train,val,test}.jsonl
    data/07_splits/split_manifest.json
    data/reports/stages/stage7.json

Uso:
    python -m src.data_prep.stage7_split [--seed 42]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from sklearn.model_selection import train_test_split

from src.data_prep import console, report
from src.data_prep.config import (
    LABELS,
    PROVIDERS,
    SEED,
    SPLIT_NAMES,
    SPLIT_RATIOS,
    STAGE6_BALANCED_DIR,
    STAGE7_SPLITS_DIR,
    merged_class_path,
    split_path,
)
from src.data_prep.jsonl_io import Record, read_jsonl, write_jsonl

#: Manifest do split: semente, proporções e contagens por estrato.
MANIFEST_PATH = STAGE7_SPLITS_DIR / "split_manifest.json"


def load_balanced() -> list[Record]:
    """
    Descrição:
        Carrega as duas classes balanceadas em uma única lista, na ordem fixa de
        `config.LABELS`.

    Retorna:
        Lista de registros do dataset completo.
    """
    records: list[Record] = []
    for label in LABELS:
        records.extend(read_jsonl(merged_class_path(STAGE6_BALANCED_DIR, label)))
    return records


def stratum_key(record: Record) -> str:
    """
    Descrição:
        Monta a chave de estratificação de um registro, combinando classe e
        provedor.

    Args:
        record: Registro do dataset.

    Retorna:
        Chave no formato `"<classe>|<provedor>"`, ex.: `"DD|glm"`.
    """
    return f"{record.label}|{record.source}"


def stratified_split(
    records: list[Record], ratios: tuple[float, float, float], seed: int
) -> dict[str, list[Record]]:
    """
    Descrição:
        Divide os registros em treino, validação e teste com estratificação por
        classe e provedor.

        A divisão ocorre em duas passadas. Na primeira, separa o treino do
        restante. Na segunda, divide o restante entre validação e teste, na
        proporção relativa entre eles — por exemplo, para (0.8, 0.1, 0.1) o
        restante de 20% é dividido ao meio.

    Args:
        records: Dataset completo.
        ratios: Proporções `(treino, validação, teste)`, somando 1.0.
        seed: Semente do particionamento.

    Retorna:
        Mapeamento nome da partição -> lista de registros.

    Levanta:
        ValueError: Se as proporções não somarem 1.0.
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"as proporções devem somar 1.0, recebido {ratios} = {sum(ratios)}")

    train_ratio, val_ratio, test_ratio = ratios
    strata = [stratum_key(r) for r in records]
    indices = list(range(len(records)))

    train_idx, rest_idx = train_test_split(
        indices,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        shuffle=True,
        stratify=strata,
    )

    # Proporção do teste dentro do que sobrou (para 10/10, metade).
    rest_test_fraction = test_ratio / (val_ratio + test_ratio)
    rest_strata = [strata[i] for i in rest_idx]
    val_idx, test_idx = train_test_split(
        rest_idx,
        test_size=rest_test_fraction,
        random_state=seed,
        shuffle=True,
        stratify=rest_strata,
    )

    # Ordena cada partição pelo índice original, para saída reproduzível.
    return {
        "train": [records[i] for i in sorted(train_idx)],
        "val": [records[i] for i in sorted(val_idx)],
        "test": [records[i] for i in sorted(test_idx)],
    }


def verify_disjoint(splits: dict[str, list[Record]]) -> dict[str, int]:
    """
    Descrição:
        Verifica que nenhum texto normalizado aparece em mais de uma partição.

    Args:
        splits: Partições produzidas por `stratified_split`.

    Retorna:
        Mapeamento `"a|b" -> número de textos em comum`, com zero em todos os
        pares quando não há vazamento.

    Levanta:
        AssertionError: Se houver qualquer sobreposição entre partições.
    """
    text_sets = {name: {r.text for r in records} for name, records in splits.items()}
    overlaps: dict[str, int] = {}
    names = list(splits)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            shared = text_sets[first] & text_sets[second]
            overlaps[f"{first}|{second}"] = len(shared)

    leaking = {pair: count for pair, count in overlaps.items() if count > 0}
    if leaking:
        raise AssertionError(f"vazamento entre partições detectado: {leaking}")
    return overlaps


def run(seed: int = SEED, ratios: tuple[float, float, float] = SPLIT_RATIOS) -> report.StageStats:
    """
    Descrição:
        Executa o Stage 7: carrega o dataset balanceado, particiona, verifica a
        ausência de vazamento e grava as partições e o manifest.

    Args:
        seed: Semente do particionamento.
        ratios: Proporções `(treino, validação, teste)`.

    Retorna:
        Estatísticas do estágio, já persistidas em
        `data/reports/stages/stage7.json`.
    """
    console.header("Stage 7 — split estratificado 80/10/10")

    records = load_balanced()
    console.detail(f"{len(records)} registros, {len(set(stratum_key(r) for r in records))} estratos")

    splits = stratified_split(records, ratios, seed)
    overlaps = verify_disjoint(splits)

    written: dict[str, int] = {}
    for name in SPLIT_NAMES:
        written[name] = write_jsonl(splits[name], split_path(name))

    # Contagens por partição, classe e provedor.
    per_split_label = {
        name: Counter(r.label for r in splits[name]) for name in SPLIT_NAMES
    }
    per_split_source = {
        name: Counter(r.source for r in splits[name]) for name in SPLIT_NAMES
    }

    manifest = {
        "seed": seed,
        "ratios": dict(zip(SPLIT_NAMES, ratios)),
        "stratify_by": "label+source",
        "n_total": len(records),
        "counts": written,
        "per_split_label": {n: dict(c) for n, c in per_split_label.items()},
        "per_split_source": {n: dict(c) for n, c in per_split_source.items()},
        "overlaps": overlaps,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    stats = report.StageStats(
        stage=7,
        name="split",
        totals={label: sum(per_split_label[n][label] for n in SPLIT_NAMES) for label in LABELS},
    )
    stats.extra = {
        "seed": seed,
        "ratios": dict(zip(SPLIT_NAMES, ratios)),
        "counts": written,
        "overlaps": overlaps,
    }
    stats.add_table(
        "Partições por classe",
        ["partição", *LABELS, "total", "% do total"],
        [
            [
                name,
                *[per_split_label[name][label] for label in LABELS],
                written[name],
                f"{100 * written[name] / len(records):.2f}",
            ]
            for name in SPLIT_NAMES
        ],
    )
    stats.add_table(
        "Partições por provedor",
        ["provedor", *SPLIT_NAMES, "total"],
        [
            [
                provider,
                *[per_split_source[name][provider] for name in SPLIT_NAMES],
                sum(per_split_source[name][provider] for name in SPLIT_NAMES),
            ]
            for provider in PROVIDERS
        ],
    )
    stats.note(
        f"estratificação por classe e provedor ({len(set(stratum_key(r) for r in records))} estratos), "
        f"semente {seed}"
    )
    stats.note(
        "interseção de texto entre partições: "
        + "  ".join(f"{pair}={count}" for pair, count in overlaps.items())
        + " (zero confirma ausência de vazamento)"
    )
    stats.render_console()
    report.save_stage_stats(stats)
    return stats


def main(argv: list[str] | None = None) -> int:
    """
    Descrição:
        Ponto de entrada de linha de comando do Stage 7.

    Args:
        argv: Argumentos de linha de comando. `None` usa `sys.argv`.

    Retorna:
        Código de saída do processo (0 em sucesso).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--seed", type=int, default=SEED, help=f"semente do particionamento (padrão: {SEED})"
    )
    args = parser.parse_args(argv)
    run(seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
