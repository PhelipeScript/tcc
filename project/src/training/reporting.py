"""
Descrição:
    Geração dos artefatos de uma execução de treino.

    Grava tudo o que o capítulo de resultados do TCC precisa: métricas em JSON,
    matriz de confusão (CSV e PNG), curvas ROC e DET com o ponto de EER
    marcado, predições por exemplo, métricas desagregadas por provedor e um
    manifest com configuração, hardware e versões — para que a execução seja
    auditável meses depois.

    O `matplotlib` é importado dentro das funções que desenham, de modo que a
    ausência dele degrade apenas os gráficos, sem impedir o treino nem a
    gravação das métricas.

Uso:
    from src.training import reporting
    reporting.save_metrics(metrics, run_dir, "test")
"""

from __future__ import annotations

import json
import platform
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.data_prep import console
from src.training.config import LABEL_NAMES
from src.training.metrics import classification_metrics, confusion

#: Ordem canônica das métricas nas tabelas comparativas do TCC.
REPORT_METRICS: tuple[str, ...] = (
    "accuracy",
    "precision_dd",
    "recall_dd",
    "f1_dd",
    "f1_macro",
    "roc_auc",
    "eer",
)


def save_json(payload: Any, path: Path) -> Path:
    """
    Descrição:
        Grava um objeto em JSON indentado, criando o diretório pai.

    Args:
        payload: Objeto serializável.
        path: Caminho de destino.

    Retorna:
        O caminho gravado.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def save_metrics(metrics: dict[str, float], run_dir: Path, split: str) -> Path:
    """
    Descrição:
        Grava as métricas de uma partição.

    Args:
        metrics: Métricas calculadas.
        run_dir: Diretório da execução.
        split: Nome da partição (`"val"` ou `"test"`).

    Retorna:
        Caminho do arquivo gravado.
    """
    return save_json(metrics, run_dir / f"metrics_{split}.json")


def log_metrics(metrics: dict[str, float], title: str) -> None:
    """
    Descrição:
        Imprime as métricas no terminal, em ordem canônica.

    Args:
        metrics: Métricas calculadas.
        title: Título do bloco.
    """
    console.header(title)
    console.table(
        ["métrica", "valor"],
        [[key, f"{metrics[key]:.4f}"] for key in REPORT_METRICS if key in metrics],
    )


def save_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, run_dir: Path, split: str
) -> tuple[Path, Path | None]:
    """
    Descrição:
        Grava a matriz de confusão em CSV e, quando o matplotlib estiver
        disponível, também em PNG com as contagens anotadas.

    Args:
        y_true: Rótulos verdadeiros.
        y_pred: Rótulos preditos.
        run_dir: Diretório da execução.
        split: Nome da partição.

    Retorna:
        Tupla `(caminho_csv, caminho_png_ou_None)`.
    """
    matrix = confusion(y_true, y_pred)
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / f"confusion_matrix_{split}.csv"
    header = "verdadeiro/predito," + ",".join(LABEL_NAMES)
    lines = [header] + [
        f"{LABEL_NAMES[i]}," + ",".join(str(int(v)) for v in matrix[i])
        for i in range(len(LABEL_NAMES))
    ]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    png_path: Path | None = None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(4.2, 3.8))
        axis.imshow(matrix, cmap="Blues")
        axis.set_xticks(range(len(LABEL_NAMES)), LABEL_NAMES)
        axis.set_yticks(range(len(LABEL_NAMES)), LABEL_NAMES)
        axis.set_xlabel("predito")
        axis.set_ylabel("verdadeiro")
        axis.set_title(f"Matriz de confusão — {split}")
        limit = matrix.max() / 2.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                axis.text(
                    j, i, f"{int(matrix[i, j]):,}".replace(",", "."),
                    ha="center", va="center",
                    color="white" if matrix[i, j] > limit else "black",
                )
        figure.tight_layout()
        png_path = run_dir / f"confusion_matrix_{split}.png"
        figure.savefig(png_path, dpi=150)
        plt.close(figure)
    except ImportError:
        console.warn("matplotlib indisponível: matriz de confusão gravada só em CSV")

    return csv_path, png_path


def save_curves(
    y_true: np.ndarray, y_score: np.ndarray, run_dir: Path, split: str, eer: float
) -> list[Path]:
    """
    Descrição:
        Grava as curvas ROC e DET da partição, marcando o ponto de EER.

        A curva DET (falsos positivos contra falsos negativos) é a leitura
        natural para um sistema de ativação: mostra o compromisso entre acordar
        indevidamente e deixar de acordar quando devia.

    Args:
        y_true: Rótulos verdadeiros.
        y_score: Score contínuo da classe positiva.
        run_dir: Diretório da execução.
        split: Nome da partição.
        eer: EER calculado, para anotar no gráfico.

    Retorna:
        Lista dos caminhos gravados (vazia se o matplotlib não estiver
        disponível ou os rótulos tiverem uma única classe).
    """
    if len(np.unique(y_true)) < 2:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_auc_score, roc_curve
    except ImportError:
        console.warn("matplotlib indisponível: curvas ROC/DET não foram geradas")
        return []

    fpr, tpr, _ = roc_curve(y_true, y_score)
    saved: list[Path] = []

    figure, axis = plt.subplots(figsize=(4.4, 4.0))
    axis.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_score):.4f}")
    axis.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
    axis.set_xlabel("taxa de falsos positivos")
    axis.set_ylabel("taxa de verdadeiros positivos")
    axis.set_title(f"Curva ROC — {split}")
    axis.legend(loc="lower right")
    figure.tight_layout()
    path = run_dir / f"roc_curve_{split}.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    saved.append(path)

    figure, axis = plt.subplots(figsize=(4.4, 4.0))
    axis.plot(fpr, 1.0 - tpr, label="DET")
    if not np.isnan(eer):
        axis.plot([eer], [eer], "o", color="crimson", label=f"EER = {eer:.4f}")
    axis.set_xlabel("taxa de falsos positivos (acordar sem querer)")
    axis.set_ylabel("taxa de falsos negativos (não acordar)")
    axis.set_title(f"Curva DET — {split}")
    axis.legend(loc="upper right")
    figure.tight_layout()
    path = run_dir / f"det_curve_{split}.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    saved.append(path)

    return saved


def save_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    sources: Sequence[str],
    run_dir: Path,
    split: str,
) -> Path:
    """
    Descrição:
        Grava as predições por exemplo, em JSONL.

        É a base para análise de erro, para o teste de McNemar entre modelos e
        para explicações de interpretabilidade posteriores.

    Args:
        y_true: Rótulos verdadeiros.
        y_pred: Rótulos preditos.
        y_score: Score contínuo da classe positiva.
        sources: Provedor de origem de cada exemplo.
        run_dir: Diretório da execução.
        split: Nome da partição.

    Retorna:
        Caminho do arquivo gravado.
    """
    path = run_dir / f"predictions_{split}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for true, pred, score, source in zip(y_true, y_pred, y_score, sources, strict=True):
            handle.write(
                json.dumps(
                    {
                        "label": LABEL_NAMES[int(true)],
                        "pred": LABEL_NAMES[int(pred)],
                        "score_dd": round(float(score), 6),
                        "source": source,
                        "correto": bool(int(true) == int(pred)),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


def save_per_source_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    sources: Sequence[str],
    run_dir: Path,
    split: str,
) -> Path:
    """
    Descrição:
        Calcula e grava as métricas separadas por provedor de origem.

        Serve para verificar se o modelo aprendeu a distinguir DD de NDD ou
        apenas o estilo de um gerador específico: variação grande entre
        provedores é sinal de que a fonte está sendo usada como atalho.

    Args:
        y_true: Rótulos verdadeiros.
        y_pred: Rótulos preditos.
        y_score: Score contínuo da classe positiva.
        sources: Provedor de origem de cada exemplo.
        run_dir: Diretório da execução.
        split: Nome da partição.

    Retorna:
        Caminho do arquivo gravado.
    """
    sources_array = np.asarray(sources)
    per_source: dict[str, Any] = {}
    skipped: list[str] = []

    for provider in sorted(set(sources)):
        mask = sources_array == provider
        subset_true = y_true[mask]
        # Um provedor com uma só classe no subconjunto não tem ROC nem EER
        # definidos. Acontece em subamostras pequenas (testes de fumaça); no
        # conjunto de teste completo todos os provedores têm as duas classes.
        if len(np.unique(subset_true)) < 2:
            skipped.append(provider)
            per_source[provider] = {
                "n": int(mask.sum()),
                "observacao": "classe única no subconjunto; métricas não calculadas",
            }
            continue
        per_source[provider] = {
            "n": int(mask.sum()),
            **classification_metrics(subset_true, y_pred[mask], y_score[mask]),
        }

    if skipped:
        console.warn(
            f"métricas por provedor omitidas para {', '.join(skipped)}: classe única no subconjunto"
        )
    return save_json(per_source, run_dir / f"metrics_by_source_{split}.json")


def save_error_samples(
    texts: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    run_dir: Path,
    split: str,
    limit: int = 100,
) -> Path:
    """
    Descrição:
        Grava os erros mais confiantes do modelo, separados em falsos positivos
        e falsos negativos, ordenados pela confiança da predição errada.

        São o material qualitativo mais útil para discutir a fronteira DD/NDD no
        texto do TCC.

    Args:
        texts: Textos dos exemplos avaliados.
        y_true: Rótulos verdadeiros.
        y_pred: Rótulos preditos.
        y_score: Score contínuo da classe positiva.
        run_dir: Diretório da execução.
        split: Nome da partição.
        limit: Número máximo de erros gravados por tipo.

    Retorna:
        Caminho do arquivo gravado.
    """
    wrong = np.nonzero(y_true != y_pred)[0]
    false_positive = [i for i in wrong if y_true[i] == 0]
    false_negative = [i for i in wrong if y_true[i] == 1]
    false_positive.sort(key=lambda i: -y_score[i])
    false_negative.sort(key=lambda i: y_score[i])

    payload = {
        "falsos_positivos": [
            {"text": texts[i], "score_dd": round(float(y_score[i]), 6)}
            for i in false_positive[:limit]
        ],
        "falsos_negativos": [
            {"text": texts[i], "score_dd": round(float(y_score[i]), 6)}
            for i in false_negative[:limit]
        ],
        "total_falsos_positivos": len(false_positive),
        "total_falsos_negativos": len(false_negative),
    }
    return save_json(payload, run_dir / f"errors_{split}.json")


def git_sha() -> str | None:
    """
    Descrição:
        Obtém o SHA do commit atual, para registrar no manifest da execução.

    Retorna:
        SHA do `HEAD`, ou `None` se não estiver em um repositório git.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def package_versions() -> dict[str, str]:
    """
    Descrição:
        Coleta as versões dos pacotes que influenciam o resultado do treino.

    Retorna:
        Dicionário `pacote -> versão`. Pacotes ausentes ficam com `"ausente"`.
    """
    versions: dict[str, str] = {"python": platform.python_version()}
    for name in ("torch", "transformers", "accelerate", "datasets", "sklearn", "numpy"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "desconhecida")
        except ImportError:
            versions[name] = "ausente"
    return versions


def save_run_manifest(payload: dict[str, Any], run_dir: Path) -> Path:
    """
    Descrição:
        Grava o manifest da execução, acrescentando git SHA, versões de pacotes
        e informações da plataforma.

    Args:
        payload: Conteúdo específico da execução (configuração, hardware,
            métricas, tempos).
        run_dir: Diretório da execução.

    Retorna:
        Caminho do manifest gravado.
    """
    payload = dict(payload)
    payload["git_sha"] = git_sha()
    payload["packages"] = package_versions()
    payload["platform"] = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    }
    return save_json(payload, run_dir / "run_manifest.json")


def build_comparison(runs: dict[str, dict[str, float]]) -> tuple[str, str]:
    """
    Descrição:
        Monta a tabela comparativa entre modelos, em Markdown e em LaTeX.

        É o artefato final do experimento: uma linha por modelo, com as métricas
        na ordem de `REPORT_METRICS`.

    Args:
        runs: Mapeamento `nome do modelo -> métricas do conjunto de teste`.

    Retorna:
        Tupla `(markdown, latex)`.
    """
    header = ["modelo", *REPORT_METRICS]
    rows = [
        [name, *[f"{metrics.get(key, float('nan')):.4f}" for key in REPORT_METRICS]]
        for name, metrics in runs.items()
    ]

    markdown = [
        "# Comparação entre modelos — conjunto de teste",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    markdown += ["| " + " | ".join(row) + " |" for row in rows]
    markdown.append("")

    latex = [
        "\\begin{table}[htb]",
        "\\centering",
        "\\caption{Desempenho dos modelos no conjunto de teste.}",
        "\\label{tab:resultados}",
        "\\begin{tabular}{l" + "r" * len(REPORT_METRICS) + "}",
        "\\hline",
        " & ".join(h.replace("_", "\\_") for h in header) + " \\\\",
        "\\hline",
    ]
    latex += [" & ".join(row) + " \\\\" for row in rows]
    latex += ["\\hline", "\\end{tabular}", "\\end{table}", ""]

    return "\n".join(markdown), "\n".join(latex)
