"""
Descrição:
    Pipeline ASR simulado (TTS → Whisper) para os objetivos específicos 2
    (estratégia n-best) e 3 (impacto do WER) do TCC1, sem depender de áudio
    real gravado — que não existe para as frases sintéticas geradas por LLM.

    Sintetiza áudio de uma amostra do conjunto de teste via TTS, transcreve
    com `faster-whisper` (extraindo n-best real via beam search, não uma
    aproximação), calcula o WER contra o texto original e roda os
    classificadores já treinados sobre texto original / 1-best / n-best,
    medindo a degradação em função do erro de transcrição.

Uso:
    python -m src.asr_eval.run_asr_eval
"""

from __future__ import annotations

__all__ = [
    "config",
    "sampling",
    "synth",
    "transcribe_nbest",
    "wer",
    "wer_impact",
    "classify",
    "engine",
]
