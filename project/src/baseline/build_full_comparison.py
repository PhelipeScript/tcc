"""
Descrição:
    Monta a tabela comparativa completa — os 3 Transformers e os 2 baselines
    clássicos — a partir dos `metrics_test.json` já gravados por cada
    execução. Não treina nada; só consolida o que já existe em `outputs/`.

    Mantido separado de `outputs/comparison.md` (gerado por
    `src.training.train_all`, só com os 3 Transformers) para não sobrescrever
    esse artefato: `comparison_full.md` é o que entra no capítulo de
    Resultados do TCC, `comparison.md` continua sendo o resumo interno do
    treino dos Transformers.

Uso:
    uv run python -m src.baseline.build_full_comparison
"""

from __future__ import annotations

import json

from src.data_prep import console
from src.training import reporting
from src.training.config import DEFAULT_OUTPUT_DIR

#: Ordem de exibição na tabela final: Transformers primeiro, baselines depois.
RUN_NAMES: tuple[str, ...] = (
    "bertimbau",
    "albertina",
    "debertinha",
    "baseline_svm",
    "baseline_logreg",
)


def main() -> int:
    """
    Descrição:
        Lê `metrics_test.json` de cada execução listada em `RUN_NAMES` e
        grava `outputs/comparison_full.{md,tex,json}`. Execuções ausentes são
        avisadas e omitidas da tabela, em vez de interromper o script.

    Retorna:
        Código de saída do processo (sempre 0).
    """
    runs: dict[str, dict[str, float]] = {}
    for name in RUN_NAMES:
        path = DEFAULT_OUTPUT_DIR / name / "metrics_test.json"
        if not path.exists():
            console.warn(f"{name}: {path} não encontrado, omitido da comparação")
            continue
        runs[name] = json.loads(path.read_text(encoding="utf-8"))

    if not runs:
        console.error("nenhuma execução encontrada em outputs/; nada a comparar")
        return 1

    markdown, latex = reporting.build_comparison(runs)
    console.header("comparação completa — Transformers + baselines")
    print(markdown)

    reporting.save_json(runs, DEFAULT_OUTPUT_DIR / "comparison_full.json")
    (DEFAULT_OUTPUT_DIR / "comparison_full.md").write_text(markdown, encoding="utf-8")
    (DEFAULT_OUTPUT_DIR / "comparison_full.tex").write_text(latex, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
