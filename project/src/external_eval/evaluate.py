"""
Descrição:
    Roda os 3 classificadores já treinados sobre uma amostra externa
    normalizada, com rótulo NDD assumido (nenhum dos 3 corpora foi anotado
    para esta tarefa — é fala real qualquer, não comandos de voz).

    Como não há as duas classes, não há F1/ROC-AUC/EER definidos (a mesma
    razão pela qual `classification_metrics` devolve `NaN` nesses casos); a
    métrica operacional aqui é a **taxa de falso positivo**: fração dos
    exemplos que o modelo classificou como DD, isto é, quantas vezes o
    assistente "acordaria à toa" ouvindo fala real que não era dirigida a ele.

Uso:
    from src.external_eval.evaluate import load_sessions, evaluate_corpus
    sessions = load_sessions(("bertimbau", "albertina", "debertinha"))
    result = evaluate_corpus(sessions, texts)
"""

from __future__ import annotations

import numpy as np

from src.training.predict_interactive import ModelSession


def load_sessions(model_keys: tuple[str, ...]) -> dict[str, ModelSession]:
    """
    Descrição:
        Carrega o melhor checkpoint e o limiar congelado de cada modelo.

    Args:
        model_keys: Chaves dos modelos em `src.training.config.MODELS`.

    Retorna:
        Mapeamento `chave -> ModelSession`.
    """
    return {key: ModelSession.load(key) for key in model_keys}


def evaluate_corpus(
    sessions: dict[str, ModelSession], texts: list[str], *, top_false_positives: int = 30
) -> dict[str, dict]:
    """
    Descrição:
        Roda cada modelo sobre todos os textos e agrega a taxa de falso
        positivo, além dos exemplos mais confiantes classificados como DD
        (material qualitativo para a análise de erro).

    Args:
        sessions: Mapeamento `chave -> ModelSession`.
        texts: Textos já normalizados e filtrados.
        top_false_positives: Quantos falsos positivos mais confiantes manter
            por modelo.

    Retorna:
        Mapeamento `chave do modelo -> {n, n_falsos_positivos,
        taxa_falso_positivo, score_medio, score_mediano, falsos_positivos}`.
    """
    results: dict[str, dict] = {}
    for model_key, session in sessions.items():
        scores: list[float] = []
        false_positives: list[tuple[str, float]] = []
        for text in texts:
            pred, prob = session.predict(text)
            scores.append(prob)
            if pred == "DD":
                false_positives.append((text, prob))

        false_positives.sort(key=lambda pair: pair[1], reverse=True)
        n = len(texts)
        results[model_key] = {
            "n": n,
            "n_falsos_positivos": len(false_positives),
            "taxa_falso_positivo": len(false_positives) / n if n else float("nan"),
            "score_medio": float(np.mean(scores)) if scores else float("nan"),
            "score_mediano": float(np.median(scores)) if scores else float("nan"),
            "falsos_positivos": [
                {"text": text, "score_dd": round(prob, 6)}
                for text, prob in false_positives[:top_false_positives]
            ],
        }
    return results
