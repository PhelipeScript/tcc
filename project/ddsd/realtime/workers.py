"""Workers de inferência: ASR (dono único do Whisper/MLX), fuzzy e um por classificador.

Cada worker é uma thread com a própria `JobQueue` (FIFO) e carrega o próprio modelo dentro da thread: o MLX exige que o modelo seja
usado pela thread que o criou, e nenhum classificador recebe chamadas
concorrentes. A prontidão é sinalizada por `ready`, e uma falha de
carregamento fica em `error`, para o pipeline encerrar de forma limpa.
"""

from __future__ import annotations

import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ddsd.asr.types import Transcription, WhisperSegment
from ddsd.observability.logger import get_logger
from ddsd.realtime.domain import AudioTask, TaskId
from ddsd.realtime.job_queue import JobQueue, QueueClosed

log = get_logger("workers")


@dataclass(frozen=True, slots=True)
class NlpJob:
    """Entrada dos workers de NLP: identidade da tarefa + texto normalizado + metadados do Whisper."""

    task_id: TaskId
    normalized: str
    segments: tuple[WhisperSegment, ...]


class Worker(threading.Thread):
    """Base: carrega o recurso na própria thread e consome a fila até ela ser fechada e esvaziada."""

    def __init__(self, name: str, queue: JobQueue[Any]) -> None:
        super().__init__(name=name, daemon=True)
        self.queue = queue
        self.ready = threading.Event()
        self.error: BaseException | None = None
        self.busy_with: str | None = None
        self.processed = 0

    def load(self) -> None:  # pragma: no cover - implementado pelas subclasses
        raise NotImplementedError

    def handle(self, item: Any) -> None:  # pragma: no cover
        raise NotImplementedError

    def run(self) -> None:
        try:
            self.load()
        except BaseException as exc:  # noqa: BLE001 - repassado ao pipeline
            self.error = exc
            log.error("%s: falha ao carregar: %r\n%s", self.name, exc, traceback.format_exc(limit=3))
            self.ready.set()
            return
        self.ready.set()
        while True:
            try:
                item = self.queue.get(timeout=0.25)
            except QueueClosed:
                return
            if item is None:
                continue
            self.busy_with = str(item.task_id)
            try:
                self.handle(item)
            finally:
                self.busy_with = None
                self.processed += 1


class AsrWorker(Worker):
    """Transcreve cada fala completa, na ordem em que as falas fecham."""

    def __init__(self, queue: JobQueue[AudioTask], factory: Callable[[], Any],
                 on_start: Callable[[AudioTask], None],
                 on_done: Callable[[AudioTask, Transcription | None, str | None], None]) -> None:
        super().__init__("asr", queue)
        self._factory = factory
        self._on_start = on_start
        self._on_done = on_done
        self.backend: Any = None
        self.warm_up_s = 0.0

    def load(self) -> None:
        self.backend = self._factory()
        self.warm_up_s = self.backend.warm_up()
        log.info("Whisper pronto (%s, %s) - aquecimento %.1f s", self.backend.name, self.backend.model_id,
                 self.warm_up_s)

    def handle(self, task: AudioTask) -> None:
        self._on_start(task)
        try:
            result = self.backend.transcribe(task.audio)
        except Exception as exc:  # noqa: BLE001 - uma falha vira resultado com erro, nunca some
            log.error("%s: falha no Whisper: %r", task.task_id, exc)
            self._on_done(task, None, repr(exc))
            return
        self._on_done(task, result, None)


class ClassifierWorker(Worker):
    """Um classificador DD/NDD por thread (`torch.set_num_threads` limita os núcleos usados)."""

    def __init__(self, name: str, queue: JobQueue[NlpJob], factory: Callable[[], Any],
                 on_done: Callable[[TaskId, str, Any, str | None], None]) -> None:
        super().__init__(f"clf-{name}", queue)
        self.model_name = name
        self._factory = factory
        self._on_done = on_done
        self.classifier: Any = None

    def load(self) -> None:
        self.classifier = self._factory()
        self.classifier.warm_up()

    def handle(self, job: NlpJob) -> None:
        try:
            prediction = self.classifier.predict(job.normalized)
        except Exception as exc:  # noqa: BLE001
            log.error("%s: falha no %s: %r", job.task_id, self.model_name, exc)
            self._on_done(job.task_id, self.model_name, None, repr(exc))
            return
        self._on_done(job.task_id, self.model_name, prediction, None)


class FuzzyWorker(Worker):
    """Confiabilidade fuzzy dos metadados do Whisper (em paralelo aos classificadores)."""

    def __init__(self, queue: JobQueue[NlpJob], evaluate: Callable[[tuple[WhisperSegment, ...]], Any],
                 on_done: Callable[[TaskId, str, Any, str | None], None]) -> None:
        super().__init__("fuzzy", queue)
        self._evaluate = evaluate
        self._on_done = on_done

    def load(self) -> None:
        self._evaluate(())  # valida a configuração das pertinências antes de abrir a captura

    def handle(self, job: NlpJob) -> None:
        started = time.perf_counter()
        try:
            result = self._evaluate(job.segments)
        except Exception as exc:  # noqa: BLE001
            log.error("%s: falha na lógica fuzzy: %r", job.task_id, exc)
            self._on_done(job.task_id, "fuzzy", None, repr(exc))
            return
        self._on_done(job.task_id, "fuzzy", (result, time.perf_counter() - started), None)
