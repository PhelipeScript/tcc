"""
Descrição:
    Cálculo do WER (Word Error Rate) entre o texto original (falado pelo TTS)
    e a transcrição do Whisper.

    Os dois lados passam pela mesma `normalize_text` do pipeline de dados
    antes do cálculo — consistente com o resto do projeto, mas é uma
    simplificação metodológica que vale registrar no texto do TCC: WER
    "tradicional" normalmente não remove acentuação, e removê-la pode tornar
    substituições que mudam significado (`está`/`esta`) invisíveis ao WER.

Uso:
    from src.asr_eval.wer import compute_wer
    wer = compute_wer("Fecha essa aba, por favor!", "feche essa aba por favor")
"""

from __future__ import annotations

import jiwer

from src.data_prep.normalizer import normalize_text


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Descrição:
        Calcula o WER entre referência e hipótese, ambas normalizadas.

    Args:
        reference: Texto original (`text_raw` do exemplo, o que o TTS falou).
        hypothesis: Transcrição do Whisper.

    Retorna:
        WER como fração (pode exceder 1,0 se a hipótese tiver muito mais
        palavras que a referência). `nan` se a referência normalizada ficar
        vazia (WER indefinido).
    """
    reference_norm = normalize_text(reference)
    hypothesis_norm = normalize_text(hypothesis)
    if not reference_norm:
        return float("nan")
    return float(jiwer.wer(reference_norm, hypothesis_norm or " "))
