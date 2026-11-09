"""Lógica fuzzy de confiabilidade da transcrição a partir dos metadados do Whisper.

Saída: r ∈ [0, 1] e rótulo linguístico (baixa / média / alta), independentes da
classificação textual. r estima o quanto o texto do Whisper merece confiança,
como fala transcrita, e não alucinação em silêncio ou repetição degenerada.

Entradas (médias dos segmentos ponderadas pela duração):
- `avg_logprob`: confiança média dos tokens. Abaixo de -1,0 o próprio Whisper
  considera a decodificação falha (`logprob_threshold`).
- `no_speech_prob`: probabilidade de não haver fala. Acima de 0,6 o Whisper
  trata o trecho como silêncio (`no_speech_threshold`).
- `compression_ratio`: texto repetitivo comprime demais. Acima de 2,4 o Whisper
  considera a decodificação degenerada (`compression_ratio_threshold`).
- `temperature`: temperatura de fallback usada. Maior que 0 significa que a
  decodificação gulosa falhou nas checagens acima e foi refeita por amostragem.

Os pontos de quebra das pertinências estão ancorados nesses três limiares
documentados do Whisper. A validação mede se r acompanha o WER real (correlação de
Spearman nas gravações). Inferência Mamdani: E = mínimo, agregação = máximo,
defuzzificação por centroide.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ddsd.asr.types import WhisperSegment

#: Pertinências trapezoidais como polilinhas (x, y), interpoladas com np.interp.
INPUT_TERMS: dict[str, dict[str, tuple[tuple[float, ...], tuple[float, ...]]]] = {
    "avg_logprob": {
        "baixo": ((-5.0, -1.0, -0.6, 0.0), (1.0, 1.0, 0.0, 0.0)),
        "medio": ((-5.0, -1.0, -0.6, -0.35, -0.15, 0.0), (0.0, 0.0, 1.0, 1.0, 0.0, 0.0)),
        "alto": ((-5.0, -0.35, -0.15, 0.0), (0.0, 0.0, 1.0, 1.0)),
    },
    "no_speech_prob": {
        "fala": ((0.0, 0.2, 0.4, 1.0), (1.0, 1.0, 0.0, 0.0)),
        "incerto": ((0.0, 0.2, 0.4, 0.5, 0.7, 1.0), (0.0, 0.0, 1.0, 1.0, 0.0, 0.0)),
        "silencio": ((0.0, 0.5, 0.6, 1.0), (0.0, 0.0, 1.0, 1.0)),
    },
    "compression_ratio": {
        "normal": ((0.0, 1.8, 2.2, 10.0), (1.0, 1.0, 0.0, 0.0)),
        "repetitivo": ((0.0, 2.0, 2.4, 10.0), (0.0, 0.0, 1.0, 1.0)),
    },
    "temperature": {
        "nenhum": ((0.0, 0.2, 1.0), (1.0, 0.0, 0.0)),
        "fallback": ((0.0, 0.2, 1.0), (0.0, 1.0, 1.0)),
    },
}

OUTPUT_TERMS: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "baixa": ((0.0, 0.2, 0.45, 1.0), (1.0, 1.0, 0.0, 0.0)),
    "media": ((0.0, 0.25, 0.45, 0.55, 0.75, 1.0), (0.0, 0.0, 1.0, 1.0, 0.0, 0.0)),
    "alta": ((0.0, 0.55, 0.8, 1.0), (0.0, 0.0, 1.0, 1.0)),
}

#: Regras: (antecedentes {variável: termo}, consequente). Conjunção pelo mínimo.
RULES: tuple[tuple[dict[str, str], str], ...] = (
    ({"no_speech_prob": "silencio"}, "baixa"),
    ({"compression_ratio": "repetitivo"}, "baixa"),
    ({"avg_logprob": "baixo"}, "baixa"),
    ({"avg_logprob": "medio", "no_speech_prob": "incerto"}, "baixa"),
    ({"avg_logprob": "medio", "no_speech_prob": "fala", "compression_ratio": "normal"}, "media"),
    ({"avg_logprob": "alto", "no_speech_prob": "incerto"}, "media"),
    ({"temperature": "fallback"}, "media"),
    ({"avg_logprob": "alto", "no_speech_prob": "fala", "compression_ratio": "normal", "temperature": "nenhum"},
     "alta"),
)

_UNIVERSE = np.linspace(0.0, 1.0, 201)
_OUTPUT_CURVES = {term: np.interp(_UNIVERSE, xs, ys) for term, (xs, ys) in OUTPUT_TERMS.items()}

LABELS = {"baixa": "baixa", "media": "média", "alta": "alta"}


@dataclass(frozen=True, slots=True)
class ReliabilityInputs:
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float
    temperature: float

    def as_dict(self) -> dict[str, float]:
        return {"avg_logprob": self.avg_logprob, "no_speech_prob": self.no_speech_prob,
                "compression_ratio": self.compression_ratio, "temperature": self.temperature}


@dataclass(frozen=True, slots=True)
class Reliability:
    score: float
    label: str
    inputs: ReliabilityInputs | None
    rule_strengths: dict[str, float] = field(default_factory=dict)
    note: str = ""


def aggregate_inputs(segments: Sequence[WhisperSegment]) -> ReliabilityInputs | None:
    """Médias ponderadas pela duração dos segmentos (pesos iguais se as durações forem nulas)."""
    if not segments:
        return None
    weights = np.array([s.duration for s in segments], dtype=float)
    if weights.sum() <= 0:
        weights = np.ones(len(segments))

    def mean(values: list[float]) -> float:
        return float(np.average(np.asarray(values, dtype=float), weights=weights))

    return ReliabilityInputs(
        avg_logprob=mean([s.avg_logprob for s in segments]),
        no_speech_prob=mean([s.no_speech_prob for s in segments]),
        compression_ratio=mean([s.compression_ratio for s in segments]),
        temperature=mean([s.temperature for s in segments]),
    )


def membership(variable: str, term: str, value: float) -> float:
    xs, ys = INPUT_TERMS[variable][term]
    return float(np.interp(value, xs, ys))


def label_of(score: float, low_below: float, high_from: float) -> str:
    if score < low_below:
        return "baixa"
    if score >= high_from:
        return "alta"
    return "média"


def evaluate(inputs: ReliabilityInputs, *, low_below: float = 0.4, high_from: float = 0.7) -> Reliability:
    """Inferência Mamdani com defuzzificação por centroide."""
    values = inputs.as_dict()
    aggregated = np.zeros_like(_UNIVERSE)
    strengths: dict[str, float] = {}
    for antecedents, consequent in RULES:
        strength = min(membership(var, term, values[var]) for var, term in antecedents.items())
        name = " E ".join(f"{var}={term}" for var, term in antecedents.items()) + f" → {consequent}"
        strengths[name] = round(strength, 4)
        if strength > 0:
            aggregated = np.maximum(aggregated, np.minimum(strength, _OUTPUT_CURVES[consequent]))
    area = aggregated.sum()
    if area <= 0:
        return Reliability(0.0, "baixa", inputs, strengths, note="nenhuma regra ativada")
    score = float((aggregated * _UNIVERSE).sum() / area)
    return Reliability(score, label_of(score, low_below, high_from), inputs, strengths)


def reliability_of(segments: Sequence[WhisperSegment], *, low_below: float = 0.4,
                   high_from: float = 0.7) -> Reliability:
    """Confiabilidade de uma transcrição; sem segmentos (nada transcrito), é 0 (baixa)."""
    inputs = aggregate_inputs(segments)
    if inputs is None:
        return Reliability(0.0, "baixa", None, note="sem segmentos transcritos")
    return evaluate(inputs, low_below=low_below, high_from=high_from)
