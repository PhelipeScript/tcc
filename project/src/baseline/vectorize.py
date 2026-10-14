"""
Descrição:
    Construção do vetorizador TF-IDF compartilhado pelos dois baselines.

Uso:
    from src.baseline.vectorize import build_vectorizer
    vectorizer = build_vectorizer(config)
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

from src.baseline.config import BaselineConfig


def build_vectorizer(config: BaselineConfig) -> TfidfVectorizer:
    """
    Descrição:
        Monta o `TfidfVectorizer` da metodologia (n-gramas de palavra 1-3),
        com teto de vocabulário e frequência mínima de documento para manter
        o treino tratável em ~190 mil frases curtas.

        O texto de entrada já passou pela normalização do Stage 2
        (`src.data_prep.normalizer`): minúsculo, sem acento, sem pontuação.
        Por isso `lowercase=False` e o `token_pattern` padrão do
        `TfidfVectorizer` (que já ignora pontuação) são suficientes — não há
        necessidade de um pré-processador próprio.

    Args:
        config: Configuração do baseline.

    Retorna:
        `TfidfVectorizer` não ajustado (`fit` deve rodar só no treino).
    """
    return TfidfVectorizer(
        ngram_range=config.ngram_range,
        max_features=config.max_features,
        min_df=config.min_df,
        lowercase=False,
        sublinear_tf=True,
    )
