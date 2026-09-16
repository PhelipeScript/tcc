"""
Descrição:
    Normaliza e filtra as transcrições externas com as mesmas regras do
    Stage 2 do pipeline de dados (`src.data_prep.stage2_normalize`), para que
    a amostra externa passe pelo mesmo funil de qualidade do corpus sintético
    antes de chegar ao classificador.

Uso:
    from src.external_eval.normalize import normalize_and_filter
    kept = normalize_and_filter(raw_texts)
"""

from __future__ import annotations

from src.data_prep.config import MIN_CHARS, MIN_WORDS
from src.data_prep.normalizer import normalize_text


def normalize_and_filter(texts: list[str]) -> list[str]:
    """
    Descrição:
        Normaliza cada texto e descarta os que ficam menores que
        `MIN_CHARS`/`MIN_WORDS` depois de normalizados — mesmo critério do
        Stage 2, para manter a amostra externa comparável em granularidade
        ao corpus de treino.

    Args:
        texts: Transcrições brutas.

    Retorna:
        Lista de textos normalizados que passaram no filtro, sem duplicatas
        (uma ocorrência por texto normalizado distinto).
    """
    seen: set[str] = set()
    kept: list[str] = []
    for text in texts:
        normalized = normalize_text(text)
        if len(normalized) <= MIN_CHARS or len(normalized.split()) < MIN_WORDS:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        kept.append(normalized)
    return kept
