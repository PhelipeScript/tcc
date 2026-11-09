"""Gravação opcional de uma sessão (`ddsd run --gravar`): áudio de cada fala e evidências da decisão.

O painel mostra a sessão ao vivo, mas não deixa dados para análise. Com `--gravar`:
- cada fala vira `U0007.wav`: 16 kHz, PCM 16 bits, mono. É exatamente o áudio da tarefa final
  enviada ao Whisper;
- cada decisão vira uma linha de `evidencias.jsonl`, com `FinalDecision.to_dict()` e o nome do áudio;
- `sessao.json` guarda os metadados (configuração, dispositivo, versão do código) e, no fim, as
  contagens.

Assim os falsos positivos e falsos negativos reais podem ser rotulados e a sessão reproduzida offline.
Como o arquivo grava voz, a gravação só liga com a flag explícita, e o destino padrão fica em `data/`,
que não é versionado.

Concorrência: as chamadas vêm da thread do frontend (áudio) e da do agregador (decisão) e só
enfileiram. Uma thread própria escreve no disco. Nada é descartado: se o disco atrasar, a fila cresce
e um aviso é registrado (com limite de frequência).

Uso como gerenciador de contexto (`with SessionRecorder(...) as recorder:`): a sessão é fechada e o
resumo gravado em qualquer saída, inclusive quando o pipeline falha ao iniciar.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ddsd.common.io import append_jsonl, read_json, write_json
from ddsd.decision.decider import FinalDecision
from ddsd.observability.logger import get_logger
from ddsd.realtime.config import MODEL_RATE
from ddsd.realtime.domain import AudioTask

log = get_logger("sessao")

EVIDENCE_FILE = "evidencias.jsonl"
SESSION_FILE = "sessao.json"
#: A partir deste tamanho de fila, o disco está atrasado em relação à fala.
BACKLOG_WARNING = 50


class SessionRecorder:
    """Grava as falas e decisões de uma sessão num diretório próprio."""

    def __init__(self, out_dir: Path, metadata: dict[str, Any]) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        if (out_dir / EVIDENCE_FILE).exists():
            raise FileExistsError(f"{out_dir} já tem uma sessão gravada; escolha outro diretório")
        self.out_dir = out_dir
        self.started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_json(out_dir / SESSION_FILE, {"inicio": self.started, "metadados": metadata})
        self._queue: queue.Queue[tuple[str, Any] | None] = queue.Queue()
        self.audio_written = 0
        self.decisions_written = 0
        self.errors = 0
        self._last_backlog_warning = 0.0
        self._summary: dict[str, Any] | None = None
        self._thread = threading.Thread(target=self._loop, name="sessao", daemon=True)
        self._thread.start()
        log.info("gravando a sessão em %s (áudio de cada fala + evidências)", out_dir)

    # ------------------------------------------------- chamado pelo pipeline
    def utterance_audio(self, task: AudioTask) -> None:
        """Áudio da fala completa (o array da tarefa já é somente leitura; não há cópia)."""
        self._put(("audio", task))

    def decision(self, decision: FinalDecision, contexto: dict[str, Any] | None = None) -> None:
        """Decisão de uma fala, com o contexto em que ela foi processada (classificador da fala)."""
        self._put(("decisao", (decision, dict(contexto or {}))))

    def _put(self, item: tuple[str, Any]) -> None:
        self._queue.put(item)
        backlog = self._queue.qsize()
        now = time.monotonic()
        if backlog >= BACKLOG_WARNING and now - self._last_backlog_warning > 30:
            self._last_backlog_warning = now
            log.warn("gravação da sessão atrasada: %d itens na fila (disco lento?)", backlog)

    # ------------------------------------------------------- thread própria
    def _loop(self) -> None:
        while (item := self._queue.get()) is not None:
            kind, payload = item
            try:
                if kind == "audio":
                    self._write_audio(payload)
                else:
                    self._write_decision(payload)
            except Exception as exc:  # noqa: BLE001 - a gravação nunca derruba o pipeline
                self.errors += 1
                log.error("gravação da sessão falhou (%s): %r", kind, exc)

    @staticmethod
    def audio_name(utterance: int) -> str:
        return f"U{utterance:04d}.wav"

    def _write_audio(self, task: AudioTask) -> None:
        path = self.out_dir / self.audio_name(task.task_id.utterance)
        partial = path.with_name(path.name + ".parcial")
        pcm = np.clip(np.asarray(task.audio, dtype=np.float32), -1.0, 1.0)
        sf.write(partial, pcm, MODEL_RATE, subtype="PCM_16", format="WAV")
        os.replace(partial, path)
        self.audio_written += 1

    def _write_decision(self, item: tuple[FinalDecision, dict[str, Any]]) -> None:
        decision, contexto = item
        name = self.audio_name(decision.utterance)
        record = {**decision.to_dict(), "audio": name if (self.out_dir / name).exists() else None,
                  "span_amostras": [decision.span.start, decision.span.end], "contexto": contexto}
        append_jsonl(self.out_dir / EVIDENCE_FILE, [record])
        self.decisions_written += 1

    # ------------------------------------------------------------- término
    def __enter__(self) -> SessionRecorder:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self, timeout_s: float = 30.0) -> dict[str, Any]:
        """Escreve o que estiver na fila, encerra a thread e registra o resumo em `sessao.json`.

        Idempotente: chamadas seguintes devolvem o mesmo resumo sem regravar nada.
        """
        if self._summary is not None:
            return self._summary
        self._queue.put(None)
        self._thread.join(timeout_s)
        pending = self._queue.qsize()
        summary = {"fim": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "falas_gravadas": self.audio_written,
                   "decisoes_gravadas": self.decisions_written, "erros": self.errors,
                   "pendentes_no_fim": pending, "thread_encerrada": not self._thread.is_alive()}
        session = read_json(self.out_dir / SESSION_FILE)
        write_json(self.out_dir / SESSION_FILE, {**session, "resumo": summary})
        if pending or self._thread.is_alive():
            log.error("gravação da sessão encerrada com %d itens pendentes", pending)
        log.info("sessão gravada em %s: %d falas, %d decisões, %d erros", self.out_dir, self.audio_written,
                 self.decisions_written, self.errors)
        self._summary = summary
        return summary
