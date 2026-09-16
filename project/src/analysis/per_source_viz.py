"""
Descrição:
    Visualização de `metrics_by_source_test.json` — as métricas já são
    calculadas por `src.training.reporting.save_per_source_metrics` para
    todas as execuções, mas nunca tinham sido plotadas. Uma variação grande
    de desempenho entre fontes geradoras seria sinal de que o modelo aprendeu
    o estilo de um LLM gerador específico, em vez da distinção DD/NDD.

Uso:
    from src.analysis.per_source_viz import load_per_source_metric, plot_grouped
"""

from __future__ import annotations

import json
from pathlib import Path


def load_per_source_metric(metrics_path: Path, metric: str) -> dict[str, float]:
    """
    Descrição:
        Extrai uma métrica de `metrics_by_source_test.json`, ignorando fontes
        que ficaram sem a métrica calculada (classe única no subconjunto).

    Args:
        metrics_path: Caminho de `metrics_by_source_<split>.json`.
        metric: Nome da métrica (chave de `classification_metrics`).

    Retorna:
        Mapeamento `fonte -> valor`, na ordem do arquivo.
    """
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        source: values[metric]
        for source, values in payload.items()
        if metric in values
    }


def plot_grouped(
    per_model: dict[str, dict[str, float]], metric: str, output_path: Path
) -> Path:
    """
    Descrição:
        Gráfico de barras agrupadas: uma barra por (modelo, fonte), agrupada
        por fonte, para comparar a variação entre provedores de forma
        consistente entre modelos.

    Args:
        per_model: Mapeamento `nome de exibição -> {fonte: valor}`.
        metric: Nome da métrica exibida (para o rótulo do eixo Y).
        output_path: Caminho do PNG a gravar.

    Retorna:
        `output_path`.

    Levanta:
        ValueError: Se `per_model` estiver vazio.
    """
    if not per_model:
        raise ValueError("per_model vazio: nada para plotar")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    sources = sorted(set().union(*(set(values) for values in per_model.values())))
    model_names = list(per_model.keys())
    x = np.arange(len(sources))
    width = 0.8 / max(len(model_names), 1)

    figure, axis = plt.subplots(figsize=(max(6.4, 0.7 * len(sources) + 2), 4.2))
    for i, name in enumerate(model_names):
        values = [per_model[name].get(source, float("nan")) for source in sources]
        axis.bar(x + i * width, values, width=width, label=name)

    axis.set_xticks(x + width * (len(model_names) - 1) / 2, sources, rotation=30, ha="right")
    axis.set_ylabel(metric)
    axis.set_title(f"{metric} por fonte geradora — conjunto de teste")
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path
