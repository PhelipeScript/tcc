"""Configuração do pipeline em tempo real, carregada de `config/pipeline.toml` e validada."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Final, TypeVar

from ddsd.common.paths import CONFIG_DIR, resolve_path

DEFAULT_PIPELINE_CONFIG: Final[Path] = CONFIG_DIR / "pipeline.toml"

#: Taxa típica de captura do microfone; as avaliações tocam os arquivos nela para emular a captura.
CAPTURE_RATE: Final[int] = 48_000
#: Taxa do Silero VAD e do Whisper (o frontend reamostra a entrada para ela).
MODEL_RATE: Final[int] = 16_000
#: Tamanho do bloco do Silero VAD a 16 kHz.
VAD_CHUNK: Final[int] = 512
#: Transformers que o painel troca ao vivo (teclas 1-3). O pipeline usa **um** classificador por vez.
CLASSIFIER_CHOICES: Final[tuple[str, ...]] = ("bertimbau", "albertina", "debertinha")
#: Baselines TF-IDF, que também podem ser o classificador do pipeline (`[classifiers] active`).
BASELINE_CHOICES: Final[tuple[str, ...]] = ("baseline_logreg", "baseline_svm")
#: Todos os classificadores aceitos em `[classifiers] active`.
PIPELINE_CHOICES: Final[tuple[str, ...]] = CLASSIFIER_CHOICES + BASELINE_CHOICES


class PipelineConfigError(ValueError):
    """Configuração inválida."""


@dataclass(frozen=True, slots=True)
class AudioConfig:
    device: str
    samplerate: int
    latency: str
    blocksize: int
    ring_buffer_s: float
    silent_input_dbfs: float
    silent_input_s: float


@dataclass(frozen=True, slots=True)
class VadConfig:
    threshold_on: float
    threshold_off: float
    min_speech_ms: int
    min_silence_ms: int
    pre_roll_ms: int
    post_roll_ms: int


@dataclass(frozen=True, slots=True)
class SegmentationConfig:
    max_utterance_s: float


@dataclass(frozen=True, slots=True)
class AsrConfig:
    backend: str
    mlx_model: str
    faster_whisper_model: str
    faster_whisper_compute_type: str
    language: str
    temperatures: tuple[float, ...]
    compression_ratio_threshold: float
    logprob_threshold: float
    mlx_cache_limit_gb: float
    #: Semente do sorteio da nova tentativa do Whisper (mlx); torna as transcrições reprodutíveis.
    seed: int


@dataclass(frozen=True, slots=True)
class ClassifiersConfig:
    #: Classificador usado pelo pipeline (um de `PIPELINE_CHOICES`); os outros são alternativas.
    active: str
    bertimbau: Path
    albertina: Path
    debertinha: Path
    baseline_logreg: Path
    baseline_svm: Path
    device: str
    torch_threads: int

    def runs(self) -> dict[str, Path]:
        """As execuções dos três Transformers (o benchmark e o painel usam estas)."""
        return {"bertimbau": self.bertimbau, "albertina": self.albertina, "debertinha": self.debertinha}

    def run_for(self, name: str) -> Path:
        """Diretório da execução de qualquer classificador aceito pelo pipeline."""
        return {**self.runs(), "baseline_logreg": self.baseline_logreg, "baseline_svm": self.baseline_svm}[name]

    @property
    def active_run(self) -> Path:
        return self.run_for(self.active)


@dataclass(frozen=True, slots=True)
class HubConfig:
    #: Repositórios do Hugging Face com os classificadores, cada um em `models/<nome do repositório>`.
    repos: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FuzzyConfig:
    low_below: float
    high_from: float


@dataclass(frozen=True, slots=True)
class DecisionConfig:
    evidence_timeout_s: float
    asr_timeout_s: float


@dataclass(frozen=True, slots=True)
class DashboardConfig:
    enabled: bool
    refresh_per_second: int
    log_lines: int


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    level: str


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    audio: AudioConfig
    vad: VadConfig
    segmentation: SegmentationConfig
    asr: AsrConfig
    classifiers: ClassifiersConfig
    hub: HubConfig
    fuzzy: FuzzyConfig
    decision: DecisionConfig
    dashboard: DashboardConfig
    logging: LoggingConfig
    source_file: Path


T = TypeVar("T")


def _section(cls: type[T], raw: dict[str, Any], name: str) -> T:
    """Instancia a seção, convertendo listas para tupla e caminhos para `Path`."""
    try:
        values = dict(raw[name])
    except KeyError as exc:
        raise PipelineConfigError(f"seção [{name}] ausente") from exc
    expected = {f.name for f in fields(cls)}
    if missing := expected - set(values):
        raise PipelineConfigError(f"[{name}] sem as chaves {sorted(missing)}")
    if extra := set(values) - expected:
        raise PipelineConfigError(f"[{name}] tem chaves desconhecidas {sorted(extra)}")
    for key, value in values.items():
        if isinstance(value, list):
            values[key] = tuple(value)
    if cls is ClassifiersConfig:
        for key in PIPELINE_CHOICES:
            values[key] = resolve_path(values[key])
    return cls(**values)


def load_pipeline_config(path: Path | None = None, **overrides: Any) -> PipelineConfig:
    """Lê, aplica sobrescritas da CLI (`secao__chave=valor`) e valida a configuração."""
    path = path or DEFAULT_PIPELINE_CONFIG
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    for dotted, value in overrides.items():
        if value is None:
            continue
        section, key = dotted.split("__", 1)
        raw.setdefault(section, {})[key] = value
    cfg = PipelineConfig(
        audio=_section(AudioConfig, raw, "audio"),
        vad=_section(VadConfig, raw, "vad"),
        segmentation=_section(SegmentationConfig, raw, "segmentation"),
        asr=_section(AsrConfig, raw, "asr"),
        classifiers=_section(ClassifiersConfig, raw, "classifiers"),
        hub=_section(HubConfig, raw, "hub"),
        fuzzy=_section(FuzzyConfig, raw, "fuzzy"),
        decision=_section(DecisionConfig, raw, "decision"),
        dashboard=_section(DashboardConfig, raw, "dashboard"),
        logging=_section(LoggingConfig, raw, "logging"),
        source_file=path,
    )
    _validate(cfg)
    return cfg


def _validate(cfg: PipelineConfig) -> None:
    v, s = cfg.vad, cfg.segmentation
    if not 0.0 < v.threshold_off < v.threshold_on < 1.0:
        raise PipelineConfigError("é preciso 0 < vad.threshold_off < vad.threshold_on < 1 (histerese)")
    if min(v.min_speech_ms, v.min_silence_ms, v.pre_roll_ms, v.post_roll_ms) < 0:
        raise PipelineConfigError("durações do VAD não podem ser negativas")
    if s.max_utterance_s < 1.0:
        raise PipelineConfigError("segmentation.max_utterance_s deve ser ≥ 1 s")
    if cfg.asr.backend not in {"mlx", "faster-whisper"}:
        raise PipelineConfigError(f"asr.backend inválido: {cfg.asr.backend!r}")
    if cfg.asr.seed < 0:
        raise PipelineConfigError("asr.seed deve ser ≥ 0")
    if cfg.classifiers.active not in PIPELINE_CHOICES:
        raise PipelineConfigError(f"classifiers.active deve ser um de {PIPELINE_CHOICES}, "
                                  f"não {cfg.classifiers.active!r}")
    if cfg.classifiers.device not in {"cpu", "mps"}:
        raise PipelineConfigError("classifiers.device deve ser 'cpu' ou 'mps'")
    names = [repo.split("/")[-1] for repo in cfg.hub.repos]
    malformed = [repo for repo in cfg.hub.repos if repo.count("/") != 1 or not all(repo.split("/"))]
    if malformed or len(set(names)) < len(names):
        raise PipelineConfigError(f"hub.repos: use <dono>/<nome>, com nomes distintos ({list(cfg.hub.repos)})")
    if not 0.0 < cfg.fuzzy.low_below < cfg.fuzzy.high_from < 1.0:
        raise PipelineConfigError("é preciso 0 < fuzzy.low_below < fuzzy.high_from < 1")
    if cfg.audio.ring_buffer_s < 1.0:
        raise PipelineConfigError("audio.ring_buffer_s deve ser ≥ 1 s")
