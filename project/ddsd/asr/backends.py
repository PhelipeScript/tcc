"""Backends Whisper large-v3-turbo: mlx-whisper (GPU Apple) e faster-whisper (CPU).

Os dois recebem os mesmos `DecodingOptions` e devolvem `Transcription`, com
metadados idênticos em estrutura. Cada instância deve ser usada por **uma única
thread**. No MLX isso é obrigatório: o modelo e o fluxo da GPU pertencem à
thread do ASR.
"""

from __future__ import annotations

import time

import numpy as np

from ddsd.asr.types import DecodingOptions, Transcription, WhisperSegment
from ddsd.observability.logger import get_logger
from ddsd.realtime.config import MODEL_RATE

log = get_logger("asr")


def _as_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError("áudio deve ser mono 1-D a 16 kHz")
    return audio


class MlxWhisperBackend:
    """mlx-whisper na GPU via Metal (fp16)."""

    name = "mlx"

    def __init__(self, model_id: str, options: DecodingOptions, cache_limit_gb: float = 0.0,
                 seed: int | None = None) -> None:
        import mlx.core as mx
        import mlx_whisper  # noqa: F401 - falha cedo se o pacote não estiver instalado

        self.model_id = model_id
        self.options = options
        self.seed = seed
        self._mx = mx
        if seed is not None:
            mx.random.seed(seed)
        if cache_limit_gb > 0:
            mx.set_cache_limit(int(cache_limit_gb * 2**30))
        device = mx.default_device()
        if "gpu" not in str(device).lower():
            log.warn("MLX está usando %s em vez da GPU", device)

    def transcribe(self, audio: np.ndarray) -> Transcription:
        import mlx_whisper

        audio = _as_audio(audio)
        started = time.perf_counter()
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model_id,
            verbose=None,  # None desliga o tqdm (False ainda desenharia a barra)
            temperature=self.options.temperatures,
            compression_ratio_threshold=self.options.compression_ratio_threshold,
            logprob_threshold=self.options.logprob_threshold,
            no_speech_threshold=None,
            condition_on_previous_text=False,
            word_timestamps=False,
            language=self.options.language,
            task="transcribe",
            fp16=True,
            best_of=1,
        )
        latency = time.perf_counter() - started
        segments = tuple(WhisperSegment(
            start=float(s["start"]), end=float(s["end"]), text=str(s["text"]),
            tokens=tuple(int(t) for t in s["tokens"]), avg_logprob=float(s["avg_logprob"]),
            compression_ratio=float(s["compression_ratio"]), no_speech_prob=float(s["no_speech_prob"]),
            temperature=float(s["temperature"]), seek=int(s["seek"])) for s in result["segments"])
        return Transcription(text_raw=result["text"], language=result["language"], segments=segments,
                             backend=self.name, model=self.model_id, audio_s=len(audio) / MODEL_RATE,
                             latency_s=latency)

    def warm_up(self) -> float:
        """Carrega o modelo e compila os kernels com 1 s de silêncio; devolve a duração."""
        started = time.perf_counter()
        self.transcribe(np.zeros(MODEL_RATE, dtype=np.float32))
        return time.perf_counter() - started

    def memory_gb(self) -> tuple[float, float]:
        """(memória ativa, pico) da GPU no MLX, em GiB."""
        return self._mx.get_active_memory() / 2**30, self._mx.get_peak_memory() / 2**30


#: Threads do faster-whisper na CPU: os 4 núcleos de desempenho do M5.
FASTER_WHISPER_THREADS = 4


class FasterWhisperBackend:
    """faster-whisper (CTranslate2) na CPU; no macOS não há aceleração por GPU.

    No M5, float32 transcreve mais rápido que int8, então é o padrão.
    """

    name = "faster-whisper"

    def __init__(self, model_id: str, options: DecodingOptions, compute_type: str = "float32",
                 cpu_threads: int = FASTER_WHISPER_THREADS) -> None:
        from faster_whisper import WhisperModel

        self.model_id = model_id
        self.options = options
        self._model = WhisperModel(model_id, device="cpu", compute_type=compute_type, cpu_threads=cpu_threads)

    def transcribe(self, audio: np.ndarray) -> Transcription:
        audio = _as_audio(audio)
        started = time.perf_counter()
        generator, info = self._model.transcribe(
            audio,
            language=self.options.language,
            task="transcribe",
            beam_size=1,
            best_of=1,
            temperature=list(self.options.temperatures),
            compression_ratio_threshold=self.options.compression_ratio_threshold,
            log_prob_threshold=self.options.logprob_threshold,
            no_speech_threshold=None,
            condition_on_previous_text=False,
            word_timestamps=False,
            vad_filter=False,  # o VAD é o Silero do próprio pipeline
        )
        raw_segments = list(generator)  # a transcrição só acontece ao consumir o gerador
        latency = time.perf_counter() - started
        segments = tuple(WhisperSegment(
            start=float(s.start), end=float(s.end), text=str(s.text), tokens=tuple(int(t) for t in s.tokens),
            avg_logprob=float(s.avg_logprob), compression_ratio=float(s.compression_ratio),
            no_speech_prob=float(s.no_speech_prob), temperature=float(s.temperature or 0.0), seek=int(s.seek))
            for s in raw_segments)
        # Mesma composição do mlx-whisper: concatenação dos textos dos segmentos, sem strip.
        text_raw = "".join(s.text for s in raw_segments)
        return Transcription(text_raw=text_raw, language=info.language, segments=segments, backend=self.name,
                             model=self.model_id, audio_s=len(audio) / MODEL_RATE, latency_s=latency,
                             language_probability=None)

    def warm_up(self) -> float:
        started = time.perf_counter()
        self.transcribe(np.zeros(MODEL_RATE, dtype=np.float32))
        return time.perf_counter() - started


def create_backend(name: str, *, mlx_model: str, faster_whisper_model: str, compute_type: str,
                   options: DecodingOptions, mlx_cache_limit_gb: float = 0.0, seed: int | None = None):
    """Instancia o backend configurado (a semente vale para o mlx, o backend do pipeline)."""
    if name == "mlx":
        return MlxWhisperBackend(mlx_model, options, mlx_cache_limit_gb, seed)
    if name == "faster-whisper":
        return FasterWhisperBackend(faster_whisper_model, options, compute_type)
    raise ValueError(f"backend desconhecido: {name!r}")
