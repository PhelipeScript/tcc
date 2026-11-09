"""Gravação guiada do conjunto de avaliação com fala real (`ddsd record-eval-set`).

Protocolo:
- 20 frases DD + 20 NDD do `text_raw` do split de teste do corpus congelado,
  estratificadas por faixa de comprimento. As mesmas frases são lidas em todos
  os cenários, o que permite comparar cenários de forma pareada.
- Cenários: (1) sala quieta, microfone interno, ~50 cm; (2) sala quieta, iPhone;
  (3) ruído de fundo real (TV, música ou ventilador), microfone interno;
  (4) ~2 m do computador, microfone interno.
- Em cada cenário também são gravados ~60 s de fala livre (conversa ou comentário,
  classe NDD) e 30 s só de ruído do ambiente (negativo do VAD).
- O cenário `conversa_interno` é o roteiro de conversa (`config/roteiro_conversa.txt`): falas NDD
  lidas como numa conversa com uma pessoa na sala, com texto de referência (dá para medir o WER).

Grava o áudio bruto na taxa do dispositivo, em WAV 16-bit, além de
`manifest.jsonl` com o texto de referência, o rótulo, o cenário, o dispositivo e
o nível.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ddsd.common.io import append_jsonl, read_jsonl
from ddsd.common.paths import CONFIG_DIR, DATA_DIR

EVAL_DIR = DATA_DIR / "eval" / "gravacoes"
#: Roteiro de conversa: uma fala NDD por linha 
CONVERSATION_SCRIPT = CONFIG_DIR / "roteiro_conversa.txt"
CONVERSATION_SCENARIO = "conversa_interno"


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    device_hint: str
    instructions: str


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("quieto_interno", "Sala quieta · microfone interno · ~50 cm", "MacBook",
             "Ambiente silencioso, sentado em frente ao MacBook, falando em tom normal."),
    Scenario("quieto_iphone", "Sala quieta · iPhone (Continuity) · ~50 cm", "iPhone",
             "Mesmo ambiente; selecione o microfone do iPhone, a ~50 cm."),
    Scenario("ruido_interno", "Ruído de fundo real · microfone interno", "MacBook",
             "Deixe TV, música ou ventilador ligado em volume de uso normal durante todo o cenário."),
    Scenario("distante_interno", "~2 m do computador · microfone interno", "MacBook",
             "Afaste-se ~2 m do MacBook (ambiente quieto) e fale em tom normal, como falaria com o assistente."),
    Scenario(CONVERSATION_SCENARIO, "Roteiro de conversa · sala quieta · microfone interno · ~50 cm", "MacBook",
             "Leia cada fala como se estivesse conversando com uma pessoa na sala (não com o computador), "
             "em tom natural."),
)


def conversation_prompts(path: Path = CONVERSATION_SCRIPT) -> list[dict[str, Any]]:
    """Falas do roteiro de conversa (todas NDD), com IDs C01, C02… na ordem do arquivo."""
    from ddsd.text.normalization import normalize_text

    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line and not line.startswith("#")]
    return [{"prompt_id": f"C{i:02d}", "text_raw": line, "text": normalize_text(line), "label": "NDD", "bin": None,
             "source": "roteiro_conversa"} for i, line in enumerate(lines, start=1)]


def select_prompts(test_split: Path, per_class: int = 20, seed: int = 7) -> list[dict[str, Any]]:
    """Frases do teste estratificadas por faixa de comprimento (mesmo conjunto em todos os cenários)."""
    rows = list(read_jsonl(test_split))
    rng = random.Random(seed)
    chosen: list[dict[str, Any]] = []
    for label in ("DD", "NDD"):
        by_bin: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            if row["label"] == label:
                by_bin.setdefault(row["bin"], []).append(row)
        bins = sorted(by_bin, key=lambda b: int(b.split("-")[0].rstrip("+")))
        picked: list[dict[str, Any]] = []
        while len(picked) < per_class:
            for b in bins:
                pool = [r for r in by_bin[b] if r not in picked]
                if pool and len(picked) < per_class:
                    picked.append(rng.choice(pool))
        chosen += picked
    rng.shuffle(chosen)
    return [{"prompt_id": f"P{i:02d}", "text_raw": r["text_raw"], "text": r["text"], "label": r["label"],
             "bin": r["bin"], "source": r["source"]} for i, r in enumerate(chosen)]


#: Perda acima disto (tempo de parede - áudio recebido) faz a sessão pedir para regravar.
MAX_LOSS_S = 0.3


@dataclass(frozen=True)
class Take:
    """Uma gravação: o áudio recebido, o tempo de parede entre começar e parar e os alertas do PortAudio."""

    audio: np.ndarray
    elapsed_s: float
    samplerate: int
    status_flags: int

    @property
    def received_s(self) -> float:
        return len(self.audio) / self.samplerate

    @property
    def lost_s(self) -> float:
        return max(0.0, self.elapsed_s - self.received_s)


class Recorder:
    """Um único stream aberto durante o cenário inteiro; grava entre `start()` e `stop()`.

    Abrir um stream por frase perdia o começo da fala no microfone do iPhone (Continuity): ocioso, ele
    leva ~2 s para voltar a entregar áudio, e esse trecho nunca chega ao callback. Com o stream
    contínuo o dispositivo fica ativo, e cada gravação compara o tempo de parede com o áudio recebido
    (`Take.lost_s`) para expor qualquer perda.
    """

    def __init__(self, device_index: int, samplerate: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.device_index = device_index
        self.samplerate = samplerate
        self._clock = clock
        self._lock = threading.Lock()
        self._chunks: list[np.ndarray] = []
        self._recording = False
        self._flags = 0
        self._started = 0.0
        self._stream: Any = None

    def __enter__(self) -> Recorder:
        import sounddevice as sd

        self._stream = sd.InputStream(device=self.device_index, channels=1, samplerate=self.samplerate,
                                      dtype="float32", latency="high", callback=self._callback)
        self._stream.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001 - assinatura do sounddevice
        with self._lock:
            if self._recording:
                self._chunks.append(indata[:, 0].copy())
                if status:
                    self._flags += 1

    def start(self) -> None:
        with self._lock:
            self._chunks, self._flags, self._recording = [], 0, True
            self._started = self._clock()

    def stop(self) -> Take:
        elapsed = self._clock() - self._started
        if self._stream is not None:  # deixa chegar o áudio já capturado que ainda está no buffer do stream
            time.sleep(min(0.5, float(self._stream.latency) + 0.05))
        with self._lock:
            self._recording = False
            chunks, flags, self._chunks = self._chunks, self._flags, []
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        return Take(audio, elapsed, self.samplerate, flags)

    def record(self, *, seconds: float | None = None, stop_event: threading.Event | None = None) -> Take:
        self.start()
        if seconds is not None:
            time.sleep(seconds)
        elif stop_event is not None:
            stop_event.wait()
        return self.stop()


def _save(path: Path, audio: np.ndarray, rate: int) -> dict[str, float]:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, rate, subtype="PCM_16")
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    rms = float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0
    return {"duration_s": round(len(audio) / rate, 3), "peak_dbfs": round(20 * np.log10(max(peak, 1e-9)), 1),
            "rms_dbfs": round(20 * np.log10(max(rms, 1e-9)), 1)}


def run_session(test_split: Path, scenario_keys: list[str] | None = None) -> int:
    """Sessão interativa: cenário por cenário, com retomada (o que já foi gravado é pulado)."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt

    from ddsd.audio.capture import list_input_devices, resolve_device

    console = Console()
    prompts = select_prompts(test_split)
    manifest = EVAL_DIR / "manifest.jsonl"
    done = {(r["scenario"], r["item"]) for r in read_jsonl(manifest)} if manifest.exists() else set()
    scenarios = [s for s in SCENARIOS if not scenario_keys or s.key in scenario_keys]
    console.print(Panel(f"{len(prompts)} frases por cenário + 60 s de fala livre + 30 s só de ruído; no roteiro "
                        f"de conversa, {len(conversation_prompts())} falas.\n"
                        "Enter começa e Enter termina cada gravação. Fale naturalmente, como falaria com o "
                        "assistente (DD) ou com uma pessoa (NDD), conforme o rótulo indicado.",
                        title="Gravação do conjunto de avaliação"))
    for scenario in scenarios:
        console.rule(f"[bold]{scenario.title}")
        console.print(scenario.instructions)
        devices = list_input_devices()
        for d in devices:
            console.print(f"  [{d.index}] {d.name}")
        spec = Prompt.ask("Dispositivo (índice ou parte do nome)", default=scenario.device_hint)
        device = resolve_device(spec)
        if not Confirm.ask(f"Gravar o cenário com {device.name}?", default=True):
            continue
        if scenario.key == CONVERSATION_SCENARIO:
            items = [(p["prompt_id"], p) for p in conversation_prompts()]
        else:
            items = [(p["prompt_id"], p) for p in prompts] + [("fala_livre", None), ("ruido", None)]
        with Recorder(device.index, device.samplerate) as recorder:
            if _record_scenario(console, Prompt, recorder, scenario, device, items, done, manifest) == "sair":
                return 0
    console.print(f"\nGravações em {EVAL_DIR}. Obrigado!")
    return 0


def _record_scenario(console: Any, prompt_cls: Any, recorder: Recorder, scenario: Scenario, device: Any,
                     items: list[tuple[str, dict[str, Any] | None]], done: set[tuple[str, str]],
                     manifest: Path) -> str:
    """Grava os itens de um cenário com o stream já aberto; devolve "sair" se o autor interromper."""
    for item_id, prompt in items:
        if (scenario.key, item_id) in done:
            continue
        path = EVAL_DIR / scenario.key / f"{item_id}.wav"
        while True:
            if item_id == "ruido":
                console.print("\n[bold]30 s só do ambiente[/bold] - não fale (mantenha o ruído do cenário).")
                prompt_cls.ask("Enter para começar", default="")
                take = recorder.record(seconds=30.0)
                label, reference = "RUIDO", ""
            else:
                if item_id == "fala_livre":
                    console.print("\n[bold]Fala livre (~60 s)[/bold]: converse ou comente algo como faria "
                                  "com alguém na sala (NDD).")
                    label, reference = "NDD", ""
                else:
                    style = "bold magenta" if prompt["label"] == "DD" else "bold cyan"
                    tone = "falando COM o computador" if prompt["label"] == "DD" else "falando com uma PESSOA"
                    console.print(f"\n[{style}]{prompt['label']}[/{style}] ({tone}) · {item_id}")
                    console.print(f"   [bold]{prompt['text_raw']}[/bold]")
                    label, reference = prompt["label"], prompt["text_raw"]
                prompt_cls.ask("Enter para gravar", default="")
                recorder.start()
                prompt_cls.ask("[red]● gravando[/red] - Enter para parar", default="")
                take = recorder.stop()
            stats = _save(path, take.audio, device.samplerate)
            stats.update(elapsed_s=round(take.elapsed_s, 3), lost_s=round(take.lost_s, 3),
                         status_flags=take.status_flags)
            console.print(f"   {stats['duration_s']} s, pico {stats['peak_dbfs']} dBFS")
            problem = take.lost_s > MAX_LOSS_S or take.status_flags > 0
            if problem:
                console.print(f"[bold yellow]   perda de áudio: o dispositivo entregou {take.received_s:.2f} s de "
                              f"{take.elapsed_s:.2f} s (alertas do PortAudio: {take.status_flags}) - "
                              "regrave (r)[/bold yellow]")
            if stats["peak_dbfs"] < -50 and item_id != "ruido":
                console.print("[yellow]   nível muito baixo - verifique o microfone/permissão[/yellow]")
            choice = prompt_cls.ask("ok / regravar / sair", choices=["o", "r", "s"], default="r" if problem else "o")
            if choice == "r":
                continue
            if choice == "s":
                return "sair"
            append_jsonl(manifest, [{"scenario": scenario.key, "item": item_id, "file": str(path.relative_to(DATA_DIR)),
                                     "label": label, "reference": reference,
                                     "bin": prompt["bin"] if prompt else None,
                                     "device": device.name, "samplerate": device.samplerate,
                                     "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **stats}])
            break
    return "ok"
