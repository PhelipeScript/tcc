"""Tipos do domínio do pipeline: identificadores, intervalos de áudio e tarefas.

Toda mensagem que atravessa threads é imutável (`frozen=True`). O áudio de uma
tarefa é uma cópia somente leitura (`writeable=False`), de modo que nenhuma
thread consegue alterar o que outra está processando.

Identificadores:
- fala: `U0007`;
- processamento da fala completa: `U0007-F`.

Intervalos de áudio (`AudioSpan`) usam **amostras absolutas a 16 kHz** contadas
desde o início da captura, o que permite cruzar qualquer resultado com o áudio
exato que o gerou.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np

from ddsd.realtime.config import MODEL_RATE


@dataclass(frozen=True, slots=True)
class AudioSpan:
    """Intervalo [start, end) em amostras absolutas a 16 kHz."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.end < self.start or self.start < 0:
            raise ValueError(f"intervalo inválido: {self.start}-{self.end}")

    @property
    def samples(self) -> int:
        return self.end - self.start

    @property
    def duration_s(self) -> float:
        return self.samples / MODEL_RATE

    def seconds(self) -> tuple[float, float]:
        """(início, fim) em segundos desde o início da captura."""
        return self.start / MODEL_RATE, self.end / MODEL_RATE

    def __str__(self) -> str:
        a, b = self.seconds()
        return f"{a:.2f}-{b:.2f}s"


@dataclass(frozen=True, slots=True, order=True)
class TaskId:
    """Identidade da tarefa de uma fala: o processamento da fala completa (`U0007-F`)."""

    utterance: int

    @property
    def utterance_label(self) -> str:
        return f"U{self.utterance:04d}"

    def __str__(self) -> str:
        return f"{self.utterance_label}-F"


class CloseReason(enum.StrEnum):
    END_OF_SPEECH = "fim_da_fala"
    MAX_DURATION = "limite_maximo"
    SHUTDOWN = "encerramento"
    ERROR = "erro"
    PAUSE = "pausa"  # microfone pausado pelo usuário no painel


def _readonly(audio: np.ndarray) -> np.ndarray:
    copy = np.array(audio, dtype=np.float32, copy=True)
    copy.flags.writeable = False
    return copy


@dataclass(frozen=True, slots=True, eq=False)
class AudioTask:
    """Áudio de uma fala completa a transcrever, com a origem temporal."""

    task_id: TaskId
    span: AudioSpan
    audio: np.ndarray
    #: Instante (time.monotonic) em que a última amostra do intervalo foi capturada.
    captured_at: float
    created_at: float
    close_reason: CloseReason | None = None
    continuation_of: int | None = None

    @classmethod
    def create(cls, task_id: TaskId, span: AudioSpan, audio: np.ndarray, captured_at: float,
               created_at: float, close_reason: CloseReason | None = None,
               continuation_of: int | None = None) -> AudioTask:
        if len(audio) != span.samples:
            raise ValueError(f"{task_id}: áudio com {len(audio)} amostras para intervalo de {span.samples}")
        return cls(task_id, span, _readonly(audio), captured_at, created_at, close_reason, continuation_of)


@dataclass(frozen=True, slots=True)
class UtteranceOpened:
    """Evento do segmentador: início de uma fala."""

    utterance: int
    start_sample: int
    opened_at: float
    continuation_of: int | None = None


@dataclass(frozen=True, slots=True)
class UtteranceClosed:
    """Evento do segmentador: a fala terminou e sua tarefa final está pronta."""

    task: AudioTask
    #: Instante de captura da última amostra com fala (referência da latência de decisão).
    speech_ended_at: float
    extra: dict[str, object] = field(default_factory=dict)
