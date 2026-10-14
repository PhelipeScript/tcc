"""
Descrição:
    Relaciona o WER por exemplo (1-best) com a taxa de acerto do
    classificador — o objetivo 3 do TCC1 ("analisar o impacto do WER no
    classificador"), além da tabela agregada por condição já gravada em
    `summary.md`.

    Lê `outputs/asr_eval/records.jsonl` (já gravado por `engine.run`) e
    agrupa os exemplos em faixas de WER, calculando a acurácia da condição
    `asr_1best` dentro de cada faixa. Como a maioria dos exemplos tem WER = 0
    (áudio sintético limpo, ambiente controlado), a faixa "0" isola o efeito
    de qualquer erro de transcrição, por menor que seja.

Uso:
    uv run python -m src.asr_eval.wer_impact
"""

from __future__ import annotations

import json
from pathlib import Path

from src.data_prep import console
from src.training.config import DEFAULT_OUTPUT_DIR, LABEL2ID

#: Faixas de WER (limite superior inclusive), rotuladas para a tabela.
WER_BUCKETS: tuple[tuple[float, str], ...] = (
    (0.0, "0 (perfeita)"),
    (0.10, "]0, 0.10]"),
    (0.30, "]0.10, 0.30]"),
    (float("inf"), "> 0.30"),
)


def bucket_for(wer: float) -> str:
    """
    Descrição:
        Encontra o rótulo da faixa de WER de um exemplo.

    Args:
        wer: WER do exemplo (1-best).

    Retorna:
        Rótulo da faixa correspondente.
    """
    for upper, label in WER_BUCKETS:
        if wer <= upper:
            return label
    return WER_BUCKETS[-1][1]


def load_records(records_path: Path) -> list[dict]:
    """
    Descrição:
        Lê `records.jsonl` do experimento de ASR simulado.

    Args:
        records_path: Caminho de `outputs/asr_eval/records.jsonl`.

    Retorna:
        Lista de registros desserializados, ignorando os sem WER definido.
    """
    records = []
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["wer_1best"] is not None:
                records.append(row)
    return records


def compute_bucket_accuracy(
    records: list[dict], model_key: str, condition: str = "asr_1best"
) -> dict[str, tuple[int, float]]:
    """
    Descrição:
        Calcula a acurácia de um modelo por faixa de WER.

    Args:
        records: Registros carregados por `load_records`.
        model_key: Modelo avaliado.
        condition: Condição cujas predições são usadas (padrão: 1-best).

    Retorna:
        Mapeamento `faixa -> (n_exemplos, accuracy)`, só com faixas não vazias.
    """
    buckets: dict[str, list[bool]] = {label: [] for _, label in WER_BUCKETS}
    for row in records:
        prediction = row["predictions"].get(condition, {}).get(model_key)
        if prediction is None:
            continue
        pred_label, _ = prediction
        correct = LABEL2ID[pred_label] == LABEL2ID[row["label"]]
        buckets[bucket_for(row["wer_1best"])].append(correct)

    return {
        label: (len(hits), sum(hits) / len(hits))
        for label, hits in buckets.items()
        if hits
    }


def main() -> int:
    """
    Descrição:
        Gera a tabela de acurácia por faixa de WER para todos os modelos
        configurados e grava `outputs/asr_eval/wer_impact.md`.

    Retorna:
        Código de saída do processo (sempre 0).
    """
    output_dir = DEFAULT_OUTPUT_DIR / "asr_eval"
    records = load_records(output_dir / "records.jsonl")
    console.info(f"{len(records)} exemplos com WER definido")

    model_keys = sorted(
        {key for row in records for key in row["predictions"].get("asr_1best", {})}
    )
    lines = [
        "# Impacto do WER na acurácia — condição asr_1best",
        "",
        "| faixa de WER | n | " + " | ".join(model_keys) + " |",
        "|---|---|" + "---|" * len(model_keys),
    ]
    for _, label in WER_BUCKETS:
        per_model = {key: compute_bucket_accuracy(records, key).get(label) for key in model_keys}
        if all(value is None for value in per_model.values()):
            continue
        n = next(value[0] for value in per_model.values() if value is not None)
        cells = [f"{value[1]:.4f}" if value else "—" for value in per_model.values()]
        lines.append(f"| {label} | {n} | " + " | ".join(cells) + " |")

    markdown = "\n".join(lines) + "\n"
    (output_dir / "wer_impact.md").write_text(markdown, encoding="utf-8")
    console.info(f"gravado em {output_dir / 'wer_impact.md'}")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
