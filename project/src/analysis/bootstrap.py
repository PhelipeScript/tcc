"""
Descrição:
    Bootstrap pareado sobre o conjunto de teste fixo, para comparar dois
    modelos com intervalo de confiança sem precisar retreinar com múltiplas
    sementes — a metodologia original previa "5 execuções com sementes
    diferentes"; como a decisão foi não retreinar, o bootstrap pareado sobre
    as mesmas 23.747 predições de teste é a alternativa estatisticamente
    aceita (Efron & Tibshirani 1993; prática comum em comparação de sistemas
    de NLP, ver Dror et al. 2018).

    "Pareado" significa que o mesmo conjunto de índices reamostrados é
    aplicado às predições dos dois modelos em cada iteração — o que só faz
    sentido se as duas listas de predições estiverem na mesma ordem de
    exemplos. Por isso `check_alignment` é obrigatória antes de interpretar
    qualquer resultado: um desalinhamento silencioso (ex.: um dos modelos
    avaliado sobre um `test.jsonl` diferente) invalidaria o teste sem dar erro.

Uso:
    from src.analysis.bootstrap import check_alignment, load_predictions_full, paired_bootstrap
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

from src.training.config import LABEL2ID
from src.training.metrics import classification_metrics

#: Métricas comparadas pelo bootstrap pareado.
BOOTSTRAP_METRICS: tuple[str, ...] = ("accuracy", "f1_macro", "eer", "roc_auc")


def load_predictions_full(predictions_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Descrição:
        Lê `predictions_test.jsonl` e monta rótulo verdadeiro, predição e
        score, todos como arrays alinhados linha a linha com o arquivo.

    Args:
        predictions_path: Caminho de `predictions_<split>.jsonl`.

    Retorna:
        Tupla `(y_true, y_pred, y_score)`.
    """
    labels: list[int] = []
    preds: list[int] = []
    scores: list[float] = []
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            labels.append(LABEL2ID[row["label"]])
            preds.append(LABEL2ID[row["pred"]])
            scores.append(row["score_dd"])
    return (
        np.asarray(labels, dtype=int),
        np.asarray(preds, dtype=int),
        np.asarray(scores, dtype=float),
    )


def check_alignment(labels_by_model: dict[str, np.ndarray]) -> None:
    """
    Descrição:
        Confirma que o rótulo verdadeiro é idêntico, linha a linha, entre
        todos os modelos informados — pré-condição do bootstrap pareado.

    Args:
        labels_by_model: Mapeamento `nome do modelo -> y_true`.

    Levanta:
        ValueError: Se dois modelos tiverem tamanhos diferentes ou rótulos
            que não coincidem linha a linha (teste desalinhado).
    """
    names = list(labels_by_model)
    reference_name, reference_labels = names[0], labels_by_model[names[0]]
    for name in names[1:]:
        labels = labels_by_model[name]
        if len(labels) != len(reference_labels):
            raise ValueError(
                f"{name!r} tem {len(labels)} exemplos, {reference_name!r} tem "
                f"{len(reference_labels)} — conjuntos de teste diferentes, "
                "bootstrap pareado inválido"
            )
        if not np.array_equal(labels, reference_labels):
            raise ValueError(
                f"predictions_test.jsonl de {name!r} não está alinhado com "
                f"{reference_name!r} (rótulos não coincidem linha a linha) — "
                "bootstrap pareado inválido"
            )


def paired_bootstrap(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    scores_a: np.ndarray,
    preds_b: np.ndarray,
    scores_b: np.ndarray,
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
    metric_keys: tuple[str, ...] = BOOTSTRAP_METRICS,
) -> dict[str, dict[str, float]]:
    """
    Descrição:
        Reamostra os índices do conjunto de teste com reposição (o mesmo
        índice para os dois modelos em cada iteração) e calcula a diferença
        `métrica(A) - métrica(B)` a cada reamostragem.

    Args:
        y_true: Rótulo verdadeiro do teste (igual para os dois modelos).
        preds_a: Predições do modelo A.
        scores_a: Scores contínuos do modelo A.
        preds_b: Predições do modelo B.
        scores_b: Scores contínuos do modelo B.
        n_bootstrap: Número de reamostragens.
        seed: Semente do gerador de números aleatórios.
        metric_keys: Métricas de `classification_metrics` a comparar.

    Retorna:
        Mapeamento `métrica -> {mean_diff, ci_low, ci_high, prob_a_better}`,
        onde `prob_a_better` é a fração de reamostragens em que A superou B
        (proxy de significância: valores perto de 0 ou 1 indicam diferença
        consistente; perto de 0,5 indica que a diferença observada pode ser
        ruído de amostragem).
    """
    n = len(y_true)
    rng = np.random.default_rng(seed)
    diffs: dict[str, list[float]] = {key: [] for key in metric_keys}

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        sample_true = y_true[indices]
        metrics_a = classification_metrics(sample_true, preds_a[indices], scores_a[indices])
        metrics_b = classification_metrics(sample_true, preds_b[indices], scores_b[indices])
        for key in metric_keys:
            value_a, value_b = metrics_a[key], metrics_b[key]
            if np.isnan(value_a) or np.isnan(value_b):
                continue
            diffs[key].append(value_a - value_b)

    result: dict[str, dict[str, float]] = {}
    for key in metric_keys:
        values = np.asarray(diffs[key], dtype=float)
        if values.size == 0:
            result[key] = {
                "mean_diff": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "prob_a_better": float("nan"),
            }
            continue
        result[key] = {
            "mean_diff": float(np.mean(values)),
            "ci_low": float(np.percentile(values, 2.5)),
            "ci_high": float(np.percentile(values, 97.5)),
            "prob_a_better": float(np.mean(values > 0.0)),
        }
    return result


def compare_all(
    predictions_by_model: dict[str, Path], *, n_bootstrap: int = 2000, seed: int = 42
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Descrição:
        Roda o bootstrap pareado entre todos os pares de modelos informados.

    Args:
        predictions_by_model: Mapeamento `nome do modelo -> caminho de
            predictions_test.jsonl`.
        n_bootstrap: Número de reamostragens por par.
        seed: Semente do bootstrap.

    Retorna:
        Mapeamento `"A_vs_B" -> resultado de paired_bootstrap`.

    Levanta:
        ValueError: Se algum par estiver desalinhado (ver `check_alignment`).
    """
    loaded = {name: load_predictions_full(path) for name, path in predictions_by_model.items()}
    check_alignment({name: values[0] for name, values in loaded.items()})

    results: dict[str, dict[str, dict[str, float]]] = {}
    for name_a, name_b in combinations(loaded, 2):
        y_true, preds_a, scores_a = loaded[name_a]
        _, preds_b, scores_b = loaded[name_b]
        results[f"{name_a}_vs_{name_b}"] = paired_bootstrap(
            y_true,
            preds_a,
            scores_a,
            preds_b,
            scores_b,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
    return results
