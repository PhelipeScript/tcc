"""Estrutura comum de transcrição e metadados, a mesma para os dois backends Whisper.

Só entram campos que os dois backends realmente fornecem por segmento:
`start`, `end`, `text`, `tokens`, `avg_logprob`, `compression_ratio`,
`no_speech_prob`, `temperature` e `seek`. Exclusividade de um backend fica
`None` no outro (ver `language_probability`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class WhisperSegment:
    """Segmento decodificado com os metadados de confiança do Whisper."""

    start: float
    end: float
    text: str
    tokens: tuple[int, ...]
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    temperature: float
    seek: int

    @property
    def duration(self) -> float:
        return max(self.end - self.start, 0.0)


@dataclass(frozen=True, slots=True)
class Transcription:
    """Resultado de uma transcrição.

    `text_raw` é exatamente o texto devolvido pelo backend, sem `strip` nem
    outra alteração; a cópia normalizada para os classificadores é feita à parte.
    """

    text_raw: str
    language: str
    segments: tuple[WhisperSegment, ...]
    backend: str
    model: str
    audio_s: float
    latency_s: float
    #: Só o faster-whisper informa; com idioma forçado ("pt") nem ele mede, então fica `None`.
    language_probability: float | None = None
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def rtf(self) -> float:
        """Fator de tempo real (latência ÷ duração do áudio)."""
        return self.latency_s / self.audio_s if self.audio_s > 0 else float("inf")


@dataclass(frozen=True, slots=True)
class DecodingOptions:
    """Parâmetros de decodificação idênticos nos dois backends.

    Decodificação gulosa, com o mesmo fallback de temperatura e os mesmos 
    limiares. `no_speech_threshold` fica desligado, para que nenhum segmento 
    seja descartado em silêncio pelo próprio Whisper: o `no_speech_prob` 
    segue para a lógica fuzzy decidir. Também não há condicionamento no texto 
    anterior, porque cada fala é transcrita de forma independente, sem vazamento 
    entre falas.
    """

    language: str = "pt"
    temperatures: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0


class WhisperBackend(Protocol):
    """Contrato dos backends: carregar uma vez e transcrever áudio mono a 16 kHz."""

    name: str
    model_id: str

    def transcribe(self, audio: np.ndarray) -> Transcription: ...

    def warm_up(self) -> float: ...
