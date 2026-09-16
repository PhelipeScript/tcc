"""
Descrição:
    Orquestra a validação externa: busca, normaliza e avalia cada corpus
    configurado, gravando os artefatos em `config.output_dir`.

Uso:
    from src.external_eval.engine import run
    run(ExternalEvalConfig())
"""

from __future__ import annotations

from src.data_prep import console
from src.external_eval.config import ExternalEvalConfig
from src.external_eval.evaluate import evaluate_corpus, load_sessions
from src.external_eval.fetch import fetch_corpus
from src.external_eval.normalize import normalize_and_filter
from src.training import reporting


def run(config: ExternalEvalConfig) -> dict[str, dict[str, dict]]:
    """
    Descrição:
        Executa a validação externa completa.

    Args:
        config: Configuração da execução.

    Retorna:
        Mapeamento `corpus -> {modelo -> resultado}` (ver
        `evaluate.evaluate_corpus`).
    """
    console.header("carregando modelos")
    sessions = load_sessions(config.model_keys)
    console.info(f"classificadores: {', '.join(config.model_keys)}")

    all_results: dict[str, dict[str, dict]] = {}
    summary_rows: list[tuple[str, str, int, int, float]] = []

    for corpus_name in config.corpora:
        console.header(f"corpus: {corpus_name}")
        raw_texts = fetch_corpus(corpus_name, config.sample_size, config.seed)
        console.info(f"{len(raw_texts)} transcrições brutas buscadas")

        texts = normalize_and_filter(raw_texts)
        console.info(f"{len(texts)} restantes após normalização e filtro de tamanho")

        result = evaluate_corpus(
            sessions, texts, top_false_positives=config.top_false_positives
        )
        all_results[corpus_name] = result

        corpus_dir = config.output_dir / corpus_name
        reporting.save_json(result, corpus_dir / "result.json")
        for model_key, values in result.items():
            console.info(
                f"  {model_key}: taxa_falso_positivo={values['taxa_falso_positivo']:.4f} "
                f"({values['n_falsos_positivos']}/{values['n']})"
            )
            summary_rows.append(
                (corpus_name, model_key, values["n"], values["n_falsos_positivos"], values["taxa_falso_positivo"])
            )

    _write_summary(config, summary_rows)
    return all_results


def _write_summary(
    config: ExternalEvalConfig, rows: list[tuple[str, str, int, int, float]]
) -> None:
    """
    Descrição:
        Grava a tabela-resumo da validação externa em Markdown.

    Args:
        config: Configuração da execução.
        rows: Linhas `(corpus, modelo, n, n_falsos_positivos, taxa)`.
    """
    lines = [
        "# Validação externa — taxa de falso positivo em fala real (rótulo NDD assumido)",
        "",
        "| corpus | modelo | n | falsos positivos | taxa |",
        "|---|---|---|---|---|",
    ]
    for corpus_name, model_key, n, n_fp, rate in rows:
        lines.append(f"| {corpus_name} | {model_key} | {n} | {n_fp} | {rate:.4f} |")
    markdown = "\n".join(lines) + "\n"
    (config.output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)
