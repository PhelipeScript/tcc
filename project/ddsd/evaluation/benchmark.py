"""Benchmarks de desempenho no Mac.

Mede:
- Whisper large-v3-turbo: latência e RTF por duração de áudio (1, 3, 5, 10 e 20 s), no mlx
  (GPU, pesos fp16) e no faster-whisper (CPU, int8 e float32, com as threads do backend). Áudio de fala
  sintetizada pelo macOS (`say`), não silêncio, para o decodificador trabalhar de verdade.
- Classificadores: latência por frase na CPU com 1, 2 e 4 threads e no MPS, um de cada vez, como no
  pipeline. O PyTorch MPS não aceita inferência concorrente de várias threads: o Metal aborta o processo.
- Frontend: custo do Silero VAD e da reamostragem (48 → 16 kHz) por segundo de áudio.
"""

from __future__ import annotations

import os
import platform
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from ddsd.common.system import code_version, power_state
from ddsd.observability.logger import get_logger

log = get_logger("benchmark")

SPEECH_TEXT = ("abre o navegador e pesquisa receitas de bolo de chocolate, depois coloca aquela playlist "
               "de estudo pra tocar baixinho enquanto eu termino o relatório da reunião de amanhã cedo. ") * 4


def speech_16k(seconds: float) -> np.ndarray:
    """Fala sintetizada pelo macOS (`say`, voz Luciana), cortada em `seconds` a 16 kHz."""
    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fala.wav"
        subprocess.run(["say", "-v", "Luciana", "-o", str(path), "--data-format=LEI16@16000", SPEECH_TEXT],
                       check=True)
        audio, rate = sf.read(str(path), dtype="float32")
    assert rate == 16_000
    need = int(seconds * 16_000)
    reps = int(np.ceil(need / len(audio)))
    return np.tile(audio, reps)[:need]


def _stats(values: list[float]) -> dict[str, float]:
    return {"n": len(values), "media_s": statistics.fmean(values), "p50_s": float(np.percentile(values, 50)),
            "p95_s": float(np.percentile(values, 95)), "min_s": min(values)}


def bench_whisper(durations: tuple[float, ...] = (1, 3, 5, 10, 20), repeats: int = 5) -> dict[str, Any]:
    from ddsd.asr.backends import FASTER_WHISPER_THREADS, FasterWhisperBackend, MlxWhisperBackend
    from ddsd.asr.types import DecodingOptions

    options = DecodingOptions()
    fw_model = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    # nome, como criar e onde roda (registrado no JSON, para a figura e a tabela dizerem o dispositivo)
    configs = [("mlx", lambda: MlxWhisperBackend("mlx-community/whisper-large-v3-turbo", options, 2.0),
                {"biblioteca": "mlx-whisper", "dispositivo": "GPU", "precisao": "fp16", "threads": None}),
               ("faster-whisper-int8", lambda: FasterWhisperBackend(fw_model, options, "int8", FASTER_WHISPER_THREADS),
                {"biblioteca": "faster-whisper", "dispositivo": "CPU", "precisao": "int8",
                 "threads": FASTER_WHISPER_THREADS}),
               ("faster-whisper-fp32", lambda: FasterWhisperBackend(fw_model, options, "float32", FASTER_WHISPER_THREADS),
                {"biblioteca": "faster-whisper", "dispositivo": "CPU", "precisao": "fp32",
                 "threads": FASTER_WHISPER_THREADS})]
    audio = {d: speech_16k(d) for d in durations}
    out: dict[str, Any] = {}
    for name, factory, where in configs:
        backend = factory()
        warm = backend.warm_up()
        rows = {}
        for d in durations:
            reps = repeats if d <= 10 else max(2, repeats // 2)
            latencies = [backend.transcribe(audio[d]).latency_s for _ in range(reps)]
            rows[f"{d:g}s"] = {**_stats(latencies), "rtf_p50": float(np.percentile(latencies, 50)) / d}
            log.info("%s %4.0f s: p50 %.2f s (RTF %.3f)", name, d, rows[f"{d:g}s"]["p50_s"], rows[f"{d:g}s"]["rtf_p50"])
        out[name] = {**where, "aquecimento_s": warm, "por_duracao": rows}
        del backend
    return out


def bench_classifiers(runs: dict[str, Path], repeats: int = 60) -> dict[str, Any]:
    import torch

    from ddsd.classifiers.model import DddClassifier

    texts = ["pausa o video", "abre o navegador e pesquisa receitas de bolo de chocolate por favor",
             "voce viu aquele video novo que a marina mandou no grupo ontem a noite", "fecha essa aba"]
    out: dict[str, Any] = {}
    for device, threads in (("cpu", 1), ("cpu", 2), ("cpu", 4), ("mps", 2)):
        if device == "mps" and not torch.backends.mps.is_available():
            continue
        torch.set_num_threads(threads)
        classifiers = {name: DddClassifier(name, path, device=device) for name, path in runs.items()}
        for clf in classifiers.values():
            clf.warm_up()
        key = f"{device}-{threads}t"
        isolated = {}
        for name, clf in classifiers.items():
            lat = [clf.predict(texts[i % len(texts)]).latency_ms / 1000 for i in range(repeats)]
            isolated[name] = _stats(lat)
        out[key] = {"isolado": isolated}
        log.info("classificadores %s: p50 (ms) %s", key, {n: round(1000 * v["p50_s"], 1) for n, v in isolated.items()})
    return out


def bench_frontend(seconds: float = 60.0) -> dict[str, Any]:
    from ddsd.audio.resampler import StreamResampler
    from ddsd.vad.silero import SileroVad

    audio16 = speech_16k(seconds)
    up = StreamResampler(16_000, 48_000)
    audio48 = np.concatenate([up.process(audio16), up.process(np.zeros(0, np.float32), last=True)])
    out = {}
    down = StreamResampler(48_000, 16_000)
    started = time.perf_counter()
    for i in range(0, len(audio48), 4800):
        down.process(audio48[i : i + 4800])
    out["reamostragem_s_por_s_audio"] = (time.perf_counter() - started) / seconds
    vad = SileroVad()
    started = time.perf_counter()
    for i in range(0, len(audio16) - 512, 512):
        vad.probability(audio16[i : i + 512])
    out["silero_s_por_s_audio"] = (time.perf_counter() - started) / seconds
    log.info("frontend por segundo de áudio: %s", {k: f"{1000 * v:.2f} ms" for k, v in out.items()})
    return out


def environment() -> dict[str, Any]:
    from importlib import metadata

    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip())
    versions = {}
    for pkg in ("mlx", "mlx-whisper", "faster-whisper", "ctranslate2", "torch", "transformers", "onnxruntime"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "ausente"
    return {"chip": chip, "memoria_gb": round(mem / 2**30, 1), "macos": platform.mac_ver()[0],
            "python": platform.python_version(), "versoes": versions,
            "carga_1min": round(os.getloadavg()[0], 2), "energia": power_state(), "codigo": code_version()}
