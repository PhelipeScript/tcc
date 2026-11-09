"""Orquestração do pipeline DDSD em tempo real.

Um classificador por vez (`[classifiers] active`: BERTimbau, Albertina, DeBERTinha ou um TF-IDF).

Ordem que evita corrida: a evidência de uma tarefa é registrada no agregador
antes de a tarefa entrar na fila. Assim, nenhum resultado chega para uma
tarefa que o agregador ainda não conhece.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import psutil

from ddsd.asr.types import DecodingOptions, Transcription
from ddsd.audio.capture import AudioCapture, CaptureError, resolve_device
from ddsd.audio.file_source import FileAudioSource
from ddsd.audio.ring_buffer import AudioRingBuffer
from ddsd.decision.decider import EvidenceAggregator, FinalDecision, TaskState
from ddsd.fuzzy.reliability import reliability_of
from ddsd.observability.logger import get_logger
from ddsd.observability.status import FeedItem, StatusBoard
from ddsd.realtime.config import BASELINE_CHOICES, MODEL_RATE, PIPELINE_CHOICES, PipelineConfig
from ddsd.realtime.domain import AudioTask, TaskId, UtteranceClosed, UtteranceOpened
from ddsd.realtime.frontend import AudioFrontend
from ddsd.realtime.job_queue import JobQueue
from ddsd.realtime.session import SessionRecorder
from ddsd.realtime.workers import AsrWorker, ClassifierWorker, FuzzyWorker, NlpJob
from ddsd.text.normalization import normalize_text
from ddsd.vad.segmenter import Event

log = get_logger("pipeline")

DecisionSink = Callable[[FinalDecision], None]


class PipelineError(RuntimeError):
    """Falha ao iniciar ou manter o pipeline."""


class Pipeline:
    """Monta, inicia e encerra todos os componentes; o áudio vem do microfone ou de arquivos."""

    def __init__(self, cfg: PipelineConfig, *, status: StatusBoard | None = None,
                 decision_sinks: list[DecisionSink] | None = None, recorder: SessionRecorder | None = None) -> None:
        self.cfg = cfg
        self.status = status or StatusBoard()
        self._sinks = list(decision_sinks or [])
        #: Gravação opcional da sessão (`--gravar`): áudio de cada fala e evidências da decisão.
        self._recorder = recorder
        self.ring: AudioRingBuffer | None = None
        self.capture: AudioCapture | None = None
        self.file_source: FileAudioSource | None = None
        self._input_rate = 0
        self.decisions: list[FinalDecision] = []
        self._stop_monitor = threading.Event()
        self._process = psutil.Process(os.getpid())
        #: O classificador do pipeline (um por vez); os outros configurados são alternativas.
        self.classifier_name = cfg.classifiers.active

        self.aggregator = EvidenceAggregator(classifier=self.classifier_name,
                                             timeout_s=cfg.decision.evidence_timeout_s,
                                             asr_timeout_s=cfg.decision.asr_timeout_s, on_decision=self._on_decision)
        self.asr_queue: JobQueue[AudioTask] = JobQueue("asr")
        self.nlp_queues: dict[str, JobQueue[NlpJob]] = {"fuzzy": self._make_nlp_queue("fuzzy")}
        options = DecodingOptions(language=cfg.asr.language, temperatures=cfg.asr.temperatures,
                                  compression_ratio_threshold=cfg.asr.compression_ratio_threshold,
                                  logprob_threshold=cfg.asr.logprob_threshold)
        self.asr_worker = AsrWorker(self.asr_queue, lambda: self._make_backend(options), self._on_asr_start,
                                    self._on_asr_done)
        self.fuzzy_worker = FuzzyWorker(
            self.nlp_queues["fuzzy"],
            lambda segments: reliability_of(segments, low_below=cfg.fuzzy.low_below, high_from=cfg.fuzzy.high_from),
            self._on_component)
        #: Um worker por classificador já usado na sessão: o ativo é carregado no início; os outros, na
        #: primeira troca pelo painel (e ficam em memória, para trocar de volta sem espera).
        self.classifier_workers: dict[str, ClassifierWorker] = {}
        self._add_classifier_worker(self.classifier_name)
        self._controls = threading.Lock()
        self._loading: str | None = None
        self._capture_paused = False
        #: Contexto de cada fala aberta (classificador da abertura), gravado com a decisão.
        self._contexts: dict[int, dict[str, object]] = {}
        self.frontend: AudioFrontend | None = None
        self._monitor = threading.Thread(target=self._monitor_loop, name="monitor", daemon=True)

    # ------------------------------------------------------------ construção
    def _make_backend(self, options: DecodingOptions):
        from ddsd.asr.backends import create_backend

        a = self.cfg.asr
        return create_backend(a.backend, mlx_model=a.mlx_model, faster_whisper_model=a.faster_whisper_model,
                              compute_type=a.faster_whisper_compute_type, options=options,
                              mlx_cache_limit_gb=a.mlx_cache_limit_gb, seed=a.seed)

    def _make_nlp_queue(self, name: str) -> JobQueue[NlpJob]:
        return JobQueue(name)

    def _add_classifier_worker(self, name: str) -> ClassifierWorker:
        queue = self._make_nlp_queue(name)
        worker = ClassifierWorker(name, queue, lambda name=name: self._make_classifier(name), self._on_component)
        self.nlp_queues[name] = queue
        self.classifier_workers[name] = worker
        return worker

    @property
    def classifier_worker(self) -> ClassifierWorker:
        """Worker do classificador ativo."""
        return self.classifier_workers[self.classifier_name]

    def _make_classifier(self, name: str):
        from ddsd.classifiers.hub import ensure

        run = ensure(self.cfg.classifiers.run_for(name), self.cfg.hub.repos)  # baixa do Hugging Face se faltar
        if name in BASELINE_CHOICES:
            from ddsd.evaluation.baselines import BaselineClassifier

            return BaselineClassifier(run)
        from ddsd.classifiers.model import DddClassifier

        return DddClassifier(name, run, device=self.cfg.classifiers.device)

    # ----------------------------------------------------------------- ciclo
    def start(self, *, input_device: str | None = None, file_rate: int | None = None) -> None:
        """Carrega os modelos (em paralelo) e então abre a entrada de áudio.

        `file_rate` liga o modo de arquivo (FileAudioSource) nessa taxa, sem abrir o microfone.
        """
        import torch

        torch.set_num_threads(self.cfg.classifiers.torch_threads)
        workers = self.workers
        a = self.cfg.asr
        session = (self._recorder.out_dir.name if self._recorder is not None
                   else time.strftime("%Y%m%d-%H%M%S"))
        self.status.set("pipeline", estado="carregando modelos", classificador=self.classifier_name,
                        execucao=self.cfg.classifiers.active_run.name, sessao=session,
                        modelo_whisper=a.mlx_model if a.backend == "mlx" else a.faster_whisper_model)
        started = time.perf_counter()
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.ready.wait()
        failed = [w for w in workers if w.error is not None]
        if failed:
            self._close_queues()
            raise PipelineError("falha ao carregar: " + "; ".join(f"{w.name}: {w.error!r}" for w in failed))
        log.info("modelos prontos em %.1f s", time.perf_counter() - started)
        threshold = getattr(self.classifier_worker.classifier, "threshold", None)
        self.status.set("pipeline", limiar=threshold, pausado=False, carregando=None)

        if file_rate is not None:
            self._input_rate = file_rate
            self.ring = AudioRingBuffer(int(self.cfg.audio.ring_buffer_s * file_rate))
            self.file_source = FileAudioSource(self.ring, file_rate, realtime=True)
            on_stall = None
            source_label = f"arquivo ({file_rate} Hz)"
            self.status.set("captura", dispositivo="arquivo (replay)", taxa=file_rate, canais=1)
        else:
            device = resolve_device(input_device if input_device is not None else self.cfg.audio.device)
            rate = self.cfg.audio.samplerate or device.samplerate
            self.ring = AudioRingBuffer(int(self.cfg.audio.ring_buffer_s * rate))
            self.capture = AudioCapture(self.ring, device, rate, self.cfg.audio.latency, self.cfg.audio.blocksize)
            self._input_rate = rate
            on_stall = self._restart_capture
            source_label = f"{device.name} ({rate} Hz)"
            self.status.set("captura", dispositivo=device.name, taxa=rate, canais=1)
        self.frontend = AudioFrontend(self.cfg, self.ring, lambda: self._input_rate, self._on_events, self.status,
                                      on_stall)
        self.frontend.start()
        self._monitor.start()
        if self.capture is not None:
            self.capture.start()
        self.status.set("captura", fonte=source_label, estado="ativa")
        self.status.set("pipeline", estado="ouvindo", backend=self.cfg.asr.backend)

    def stop(self, drain_timeout_s: float = 15.0) -> None:
        """Encerramento seguro: entrada → fala aberta vira final → finais drenadas → threads unidas."""
        self.status.set("pipeline", estado="encerrando")
        if self.capture is not None:
            self.capture.stop()
        if self.file_source is not None:
            self.file_source.stop()
        if self.frontend is not None:
            self.frontend.stop()
            self.frontend.join(5)
        deadline = time.monotonic() + drain_timeout_s
        while self.aggregator.pending() and time.monotonic() < deadline:
            self.aggregator.check_timeouts()
            time.sleep(0.05)
        if self.aggregator.pending():
            log.warn("encerramento: falas sem decisão no prazo: %s", self.aggregator.pending())
        self._close_queues()
        for worker in self.workers:
            if worker.is_alive():
                worker.join(10)
        self._stop_monitor.set()
        if self._monitor.is_alive():
            self._monitor.join(2)
        self.status.set("pipeline", estado="encerrado")

    @property
    def workers(self) -> list:
        return [self.asr_worker, self.fuzzy_worker, *list(self.classifier_workers.values())]

    def _close_queues(self) -> None:
        self.asr_queue.close()
        for queue in list(self.nlp_queues.values()):
            queue.close()

    # --------------------------------------------------- controles do painel
    @property
    def capture_paused(self) -> bool:
        return self._capture_paused

    def pause_capture(self) -> None:
        """Fecha o microfone: a fala aberta é fechada e decidida, e o resto do pipeline fica ocioso."""
        if self.capture is None or self.frontend is None or self._capture_paused:
            return
        self._capture_paused = True
        self.frontend.set_paused(True)
        self.capture.stop()
        self.status.set("captura", estado="pausada")
        self.status.set("pipeline", pausado=True)
        log.info("microfone PAUSADO pelo painel: a fala aberta é decidida e o pipeline fica ocioso")

    def resume_capture(self) -> None:
        if self.capture is None or self.frontend is None or not self._capture_paused:
            return
        try:
            self.capture.start()
        except CaptureError as exc:
            log.warn("não foi possível reabrir o microfone (%s); tentando de novo com o dispositivo atualizado", exc)
            try:
                self.capture.restart()
            except CaptureError as again:
                log.error("microfone continua pausado: %s", again)
                return
        self._input_rate = self.capture.samplerate
        self._capture_paused = False
        self.frontend.set_paused(False)
        self.status.set("captura", estado="ativa", dispositivo=self.capture.device.name, taxa=self._input_rate)
        self.status.set("pipeline", pausado=False)
        log.info("microfone RETOMADO")

    def toggle_capture(self) -> None:
        if self._capture_paused:
            self.resume_capture()
        else:
            self.pause_capture()

    def set_classifier(self, name: str) -> None:
        """Troca o classificador das próximas falas. Na primeira vez, carrega o modelo em segundo plano;
        enquanto isso, as falas seguem com o classificador atual."""
        if name not in PIPELINE_CHOICES:
            raise ValueError(f"classificador desconhecido: {name!r}")
        # Ordem das travas: agregador → controles (o `_on_decision` roda sob a do agregador). Por isso
        # nada aqui chama o agregador segurando `_controls`.
        with self._controls:
            if name == self.classifier_name:
                self._loading = None  # cancela uma troca ainda carregando
                self.status.set("pipeline", carregando=None)
                return
            if name == self._loading:
                return
            worker = self.classifier_workers.get(name)
            ready = worker is not None and worker.ready.is_set() and worker.error is None
            if not ready:
                if worker is None:
                    worker = self._add_classifier_worker(name)
                    worker.start()
                self._loading = name
        if ready:
            self._activate_classifier(name)
            return
        self.status.set("pipeline", carregando=name)
        log.info("carregando o classificador %s (%s)...", name, self.cfg.classifiers.run_for(name).name)
        threading.Thread(target=self._finish_loading, args=(name, worker), name=f"carrega-{name}",
                         daemon=True).start()

    def _finish_loading(self, name: str, worker: ClassifierWorker) -> None:
        worker.ready.wait()
        with self._controls:
            if worker.error is not None:
                log.error("falha ao carregar o classificador %s: %r", name, worker.error)
                self.classifier_workers.pop(name, None)
                queue = self.nlp_queues.pop(name, None)
                if queue is not None:
                    queue.close()
                if self._loading == name:
                    self._loading = None
                    self.status.set("pipeline", carregando=None)
                return
            if self._loading != name:  # outra troca foi pedida enquanto este carregava
                return
            self._loading = None
        self._activate_classifier(name)

    def _activate_classifier(self, name: str) -> None:
        """Ativa um classificador já carregado (sem `_controls` tomado; ver a ordem das travas)."""
        self.classifier_name = name
        self.aggregator.set_classifier(name)
        threshold = getattr(self.classifier_workers[name].classifier, "threshold", None)
        self.status.set("pipeline", classificador=name, execucao=self.cfg.classifiers.run_for(name).name,
                        limiar=threshold, carregando=None)
        log.info("classificador ativo: %s (vale a partir da próxima fala)", name)

    # --------------------------------------------------------- modo arquivo
    def play(self, signal: np.ndarray, trailing_silence_s: float = 1.5) -> tuple[int, int]:
        """Toca um sinal (na taxa do arquivo) seguido de silêncio; devolve o intervalo em amostras de 16 kHz."""
        if self.file_source is None:
            raise PipelineError("pipeline não foi iniciado em modo de arquivo")
        start16 = round(self.file_source.samples_written * MODEL_RATE / self._input_rate)
        self.file_source.play(signal)
        end16 = round(self.file_source.samples_written * MODEL_RATE / self._input_rate)
        self.file_source.play_silence(trailing_silence_s)
        return start16, end16

    def wait_idle(self, timeout_s: float = 60.0) -> bool:
        """Espera o frontend consumir o buffer e todas as falas abertas serem decididas."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            ring_empty = self.ring is None or self.ring.stats().unread == 0
            speaking = self.status.snapshot().components.get("vad", {}).get("estado") != "silêncio"
            if ring_empty and not speaking and not self.aggregator.pending():
                return True
            time.sleep(0.05)
        return False

    # -------------------------------------------------------------- eventos
    def _on_events(self, events: list[Event]) -> None:
        for event in events:
            if isinstance(event, UtteranceOpened):
                self.aggregator.utterance_opened(event.utterance, event.opened_at)
                classifier = self.aggregator.classifier_of(event.utterance)
                with self._controls:
                    self._contexts[event.utterance] = {"classificador": classifier}
                label = f"U{event.utterance:04d}"
                self.status.utterance_opened(label, event.start_sample / MODEL_RATE, time.time())
                self.status.utterance_update(label, classifier=classifier)
                self.status.incr("falas")
                cont = f" (continuação de U{event.continuation_of:04d})" if event.continuation_of else ""
                log.info("U%04d: início de fala em %.2f s%s", event.utterance, event.start_sample / MODEL_RATE, cont)
            elif isinstance(event, UtteranceClosed):
                task = event.task
                self.aggregator.utterance_closed(task.task_id, task.span, task.close_reason, event.speech_ended_at)
                self.status.utterance_update(f"U{task.task_id.utterance:04d}", state="transcrevendo",
                                             duration_s=task.span.duration_s)
                log.info("%s: fala fechada (%s, %.2f s)", task.task_id, task.close_reason.value,
                         task.span.duration_s)
                if self._recorder is not None:  # antes do ASR: o áudio é gravado antes da decisão (FIFO)
                    self._recorder.utterance_audio(task)
                self.asr_queue.put(task)

    def _on_asr_start(self, task: AudioTask) -> None:
        self.aggregator.task_state(task.task_id, TaskState.ASR)
        self.status.set("whisper", estado=f"transcrevendo {task.task_id}", inicio=time.monotonic())

    def _on_asr_done(self, task: AudioTask, result: Transcription | None, error: str | None) -> None:
        normalized = normalize_text(result.text_raw) if result is not None else None
        self.aggregator.transcription(task.task_id, result, error, normalized)
        self.status.utterance_update(f"U{task.task_id.utterance:04d}", state="analisando",
                                     final_text=None if result is None else result.text_raw,
                                     final_normalized=normalized)
        if result is None:
            self.status.incr("whisper.erros")
            self.status.set("whisper", estado="ocioso")
            return
        self.status.observe("whisper.final", result.latency_s)
        self.status.set("whisper", estado="ocioso", ultima_latencia_s=round(result.latency_s, 3),
                        ultimo_rtf=round(result.rtf, 3), ultima_tarefa=str(task.task_id))
        backend = self.asr_worker.backend
        if hasattr(backend, "memory_gb"):
            active, peak = backend.memory_gb()
            self.status.set("memoria", mlx_ativa_gb=round(active, 2), mlx_pico_gb=round(peak, 2))
        self.status.push("transcricoes", FeedItem(time.monotonic(), str(task.task_id), "final", result.text_raw,
                                                  {"rtf": round(result.rtf, 3)}))
        log.info("%s %r (%.2f s, RTF %.2f)", task.task_id, result.text_raw, result.latency_s, result.rtf)
        job = NlpJob(task.task_id, normalized or "", result.segments)
        self.nlp_queues["fuzzy"].put(job)
        if normalized:
            classifier = self.aggregator.classifier_of(task.task_id.utterance)
            queue = self.nlp_queues.get(classifier)
            if queue is not None:
                queue.put(job)
            else:  # o modelo desta fala falhou ao carregar: a tarefa fecha com erro, sem esperar o prazo
                self.aggregator.component_result(task.task_id, classifier, None, error="classificador indisponível")

    def _on_component(self, task_id: TaskId, component: str, result, error: str | None) -> None:
        kind = "final"
        if component == "fuzzy":
            reliability, seconds = result if result is not None else (None, 0.0)
            self.aggregator.component_result(task_id, "fuzzy", reliability, error)
            self.status.set("fuzzy", ultima_tarefa=str(task_id))
            if reliability is not None:
                self.status.observe("fuzzy", seconds)
                self.status.push("fuzzy", FeedItem(time.monotonic(), str(task_id), kind,
                                                   f"r={reliability.score:.2f} ({reliability.label})"))
            return
        self.aggregator.component_result(task_id, component, result, error)
        self.status.set("classificador", ultima_tarefa=str(task_id))
        if result is not None:
            self.status.observe(f"clf.{component}", result.latency_ms / 1000)
            self.status.push(f"clf.{component}", FeedItem(time.monotonic(), str(task_id), kind,
                                                          f"{result.label} p={result.p_dd:.3f}"))

    def _on_decision(self, decision: FinalDecision) -> None:
        self.decisions.append(decision)
        vote = decision.votes.get(decision.classifier)
        self.status.utterance_update(
            decision.utterance_label, state="decidida", label=decision.label, reason=decision.reason,
            p_dd=None if vote is None else vote[1], threshold=None if vote is None else vote[2],
            reliability_score=decision.reliability_score, reliability_label=decision.reliability_label,
            reliability_alert=decision.reliability_alert, latency_s=decision.latency_s,
            final_text=decision.text_raw, final_normalized=decision.normalized, duration_s=decision.span.duration_s)
        self.status.incr(f"decisoes.{decision.label}")
        if decision.latency_s is not None:
            self.status.observe("decisao", decision.latency_s)
        alert = " ⚠ confiabilidade baixa" if decision.reliability_alert else ""
        self.status.push("decisoes", FeedItem(time.monotonic(), decision.utterance_label, "final",
                                              f"{decision.label} ({decision.reason}){alert}",
                                              {"texto": decision.text_raw}))
        log.info("%s → %s (%s, %s p=%s) | r=%s | %r | %.2f s", decision.utterance_label,
                 decision.label, decision.reason, decision.classifier, None if vote is None else round(vote[1], 3),
                 decision.reliability_score, decision.text_raw, decision.latency_s or float("nan"))
        with self._controls:
            context = self._contexts.pop(decision.utterance, {})
        if self._recorder is not None:
            self._recorder.decision(decision, context)
        for sink in self._sinks:
            try:
                sink(decision)
            except Exception as exc:  # noqa: BLE001 - um consumidor com erro não derruba o pipeline
                log.error("consumidor de decisões falhou: %r", exc)

    # -------------------------------------------------------------- monitor
    def _restart_capture(self) -> None:
        if self.capture is None:
            return
        try:
            self.capture.restart()
            self._input_rate = self.capture.samplerate
            self.status.set("captura", fonte=f"{self.capture.device.name} ({self._input_rate} Hz)", estado="ativa")
        except Exception as exc:  # noqa: BLE001
            self.status.set("captura", estado="falha")
            log.error("não foi possível reabrir a captura: %r", exc)

    def activity(self) -> str:
        """"fala" (VAD em fala), "processando" (tarefas em fila, worker ocupado ou fala sem decisão),
        "pausado" (microfone fechado pelo painel) ou "ocioso" (só captura e VAD rodando)."""
        if not self._capture_paused and self.status.get("vad", "estado") == "fala":
            return "fala"
        queues = (self.asr_queue, *list(self.nlp_queues.values()))
        if (any(q.stats().pending for q in queues)
                or any(w.busy_with for w in self.workers) or self.aggregator.pending()):
            return "processando"
        return "pausado" if self._capture_paused else "ocioso"

    def _monitor_loop(self) -> None:
        self._process.cpu_percent(None)  # a primeira leitura só inicia a medição
        ticks = 0
        while not self._stop_monitor.wait(0.25):
            ticks += 1
            self.aggregator.check_timeouts()
            self.status.set("pipeline", atividade=self.activity())
            if ticks % 4 == 0:  # CPU do processo a cada ~1 s (todas as threads; pode passar de 100%)
                self.status.set("pipeline", cpu_pct=round(self._process.cpu_percent(None)))
            for name, queue in (("asr", self.asr_queue), *list(self.nlp_queues.items())):
                s = queue.stats()
                self.status.set(f"fila.{name}", pendentes=s.pending, total=s.enqueued)
            if self.ring is not None:
                r = self.ring.stats()
                self.status.set("buffer", ocupacao=round(r.fill_ratio, 3), perdidas=r.overrun_samples,
                                lido_s=round(r.read / max(self._input_rate, 1), 1))
            if self.capture is not None:
                self.status.set("captura", overflows=self.capture.overflows, callbacks=self.capture.callbacks,
                                ativa=self.capture.active)
            self.status.set("memoria", rss_gb=round(self._process.memory_info().rss / 2**30, 2))
            busy = {w.name: w.busy_with for w in self.workers}
            self.status.set("workers", **{k: (v or "ocioso") for k, v in busy.items()})
            if self.frontend is not None and self.frontend.error is not None:
                self.status.set("pipeline", estado=f"frontend falhou: {self.frontend.error!r}")


def utterances_for_file(decisions: list[FinalDecision], start16: int, end16: int) -> list[FinalDecision]:
    """Decisões cujas falas começam dentro do intervalo de um arquivo tocado no modo de arquivo."""
    return [d for d in decisions if start16 - MODEL_RATE // 2 <= d.span.start < end16]


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    from ddsd.audio.file_source import load_mono

    return load_mono(path)
