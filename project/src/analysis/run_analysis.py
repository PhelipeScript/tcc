"""
Descrição:
    Ponto de entrada da análise: gera todos os gráficos e o bootstrap
    pareado a partir dos artefatos já gravados por `src.training` e
    `src.baseline`. Não treina nada — só lê `outputs/<modelo>/`.

Uso:
    uv run python -m src.analysis.run_analysis
"""

from __future__ import annotations

import json

from src.analysis.bootstrap import compare_all
from src.analysis.calibration import load_predictions, plot_reliability_diagram
from src.analysis.config import AnalysisConfig
from src.analysis.loss_curves import load_log_history, plot_loss_curve, plot_overlay
from src.analysis.per_source_viz import load_per_source_metric, plot_grouped
from src.data_prep import console
from src.training import reporting


def run_loss_curves(config: AnalysisConfig) -> None:
    """
    Descrição:
        Gera a curva de loss individual de cada Transformer e o overlay
        comparando os três — únicos modelos que têm `trainer_state.json`
        (os baselines clássicos não usam `Trainer`).

    Args:
        config: Configuração da análise.
    """
    console.header("curvas de loss")
    histories = {}
    for model_key in config.transformer_models:
        run_dir = config.run_dir(model_key)
        if not run_dir.exists():
            console.warn(f"{model_key}: {run_dir} não encontrado, pulando")
            continue
        history = load_log_history(run_dir)
        name = config.display_name(model_key)
        histories[name] = history
        output_path = config.analysis_dir / "loss_curves" / f"{model_key}.png"
        plot_loss_curve(name, history, output_path)
        console.info(f"{model_key}: {output_path}")

    if len(histories) > 1:
        overlay_path = config.analysis_dir / "loss_curves" / "overlay.png"
        plot_overlay(histories, overlay_path)
        console.info(f"overlay: {overlay_path}")


def run_per_source(config: AnalysisConfig, metric: str = "f1_macro") -> None:
    """
    Descrição:
        Gera o gráfico de barras agrupadas de uma métrica por fonte geradora,
        para todos os modelos já treinados (Transformers e baselines).

    Args:
        config: Configuração da análise.
        metric: Métrica plotada.
    """
    console.header(f"{metric} por fonte geradora")
    per_model: dict[str, dict[str, float]] = {}
    for model_key in config.all_models:
        metrics_path = config.run_dir(model_key) / "metrics_by_source_test.json"
        if not metrics_path.exists():
            console.warn(f"{model_key}: {metrics_path} não encontrado, pulando")
            continue
        per_model[config.display_name(model_key)] = load_per_source_metric(metrics_path, metric)

    if not per_model:
        console.warn("nenhum metrics_by_source_test.json encontrado")
        return

    output_path = config.analysis_dir / "per_source" / f"{metric}_by_source.png"
    plot_grouped(per_model, metric, output_path)
    console.info(f"gravado em {output_path}")


def run_calibration(config: AnalysisConfig) -> None:
    """
    Descrição:
        Gera o diagrama de confiabilidade sobrepondo todos os modelos já
        treinados, a partir de `predictions_test.jsonl`.

    Args:
        config: Configuração da análise.
    """
    console.header("diagrama de confiabilidade")
    per_model = {}
    for model_key in config.all_models:
        predictions_path = config.run_dir(model_key) / "predictions_test.jsonl"
        if not predictions_path.exists():
            console.warn(f"{model_key}: {predictions_path} não encontrado, pulando")
            continue
        per_model[config.display_name(model_key)] = load_predictions(predictions_path)

    if not per_model:
        console.warn("nenhum predictions_test.jsonl encontrado")
        return

    output_path = config.analysis_dir / "calibration" / "reliability_diagram.png"
    plot_reliability_diagram(per_model, output_path)
    console.info(f"gravado em {output_path}")


def run_bootstrap(config: AnalysisConfig) -> None:
    """
    Descrição:
        Roda o bootstrap pareado entre todos os pares de modelos já
        treinados e grava o resultado em JSON e em uma tabela Markdown.

    Args:
        config: Configuração da análise.
    """
    console.header("bootstrap pareado")
    predictions_by_model = {
        config.display_name(model_key): config.run_dir(model_key) / "predictions_test.jsonl"
        for model_key in config.all_models
        if (config.run_dir(model_key) / "predictions_test.jsonl").exists()
    }
    if len(predictions_by_model) < 2:
        console.warn("menos de 2 modelos com predictions_test.jsonl, bootstrap pulado")
        return

    try:
        results = compare_all(
            predictions_by_model, n_bootstrap=config.n_bootstrap, seed=config.seed
        )
    except ValueError as exc:
        console.error(f"bootstrap abortado: {exc}")
        return

    output_dir = config.analysis_dir / "bootstrap"
    reporting.save_json(results, output_dir / "pairwise_bootstrap.json")

    lines = [
        "# Bootstrap pareado entre modelos — conjunto de teste",
        "",
        f"{config.n_bootstrap} reamostragens, semente {config.seed}. "
        "`prob_a_maior_que_b` perto de 0 ou 1 indica diferença consistente; "
        "perto de 0,5 indica que a diferença observada pode ser ruído de amostragem.",
        "",
        "| comparação | métrica | diff. média (A-B) | IC 95% | P(A > B) |",
        "|---|---|---|---|---|",
    ]
    for pair, metrics in results.items():
        name_a, name_b = pair.split("_vs_")
        for metric, values in metrics.items():
            lines.append(
                f"| {name_a} vs {name_b} | {metric} | {values['mean_diff']:+.4f} | "
                f"[{values['ci_low']:+.4f}, {values['ci_high']:+.4f}] | "
                f"{values['prob_a_better']:.3f} |"
            )
    markdown = "\n".join(lines) + "\n"
    (output_dir / "pairwise_bootstrap.md").write_text(markdown, encoding="utf-8")
    console.info(f"gravado em {output_dir}")
    print(markdown)


def main() -> int:
    """
    Descrição:
        Roda todas as análises disponíveis com os artefatos presentes em
        `outputs/`.

    Retorna:
        Código de saída do processo (sempre 0).
    """
    config = AnalysisConfig()
    console.info(f"modelos disponíveis: {', '.join(config.all_models) or '(nenhum)'}")
    run_loss_curves(config)
    run_per_source(config)
    run_calibration(config)
    run_bootstrap(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
