"""Configuração do corpus, carregada de `config/corpus.toml` e validada."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from ddsd.common.paths import CONFIG_DIR, resolve_path

Label = Literal["DD", "NDD"]

#: Rótulos em ordem fixa. **DD é a classe positiva** em todo o projeto.
LABELS: Final[tuple[Label, Label]] = ("DD", "NDD")

#: Origem de cada fonte: corpus inicial ou geração nova de mitigação.
ORIGINS: Final[tuple[str, str]] = ("legacy", "mitigation")

DEFAULT_CONFIG_PATH: Final[Path] = CONFIG_DIR / "corpus.toml"


class ConfigError(ValueError):
    """Configuração inválida."""


@dataclass(frozen=True, slots=True)
class LengthBin:
    """Faixa de comprimento em palavras (inclusive nas duas pontas)."""

    low: int
    high: int

    @property
    def name(self) -> str:
        return str(self.low) if self.low == self.high else f"{self.low}-{self.high}"

    def contains(self, words: int) -> bool:
        return self.low <= words <= self.high


@dataclass(frozen=True, slots=True)
class CorpusPaths:
    sources: Path
    build: Path
    frozen: Path
    taxonomy: Path
    #: Cópia do relatório e do manifesto do corpus congelado, versionável junto dos resultados.
    reports: Path
    #: Pasta do corpus inicial, lida só por `ddsd corpus import-legacy`; as fontes congeladas não dependem dela.
    legacy_dataset: Path | None = None


@dataclass(frozen=True, slots=True)
class FilterConfig:
    min_chars: int
    min_words: int
    max_words: int
    require_ascii: bool


@dataclass(frozen=True, slots=True)
class NearDedupConfig:
    enabled: bool
    threshold: float
    ngram_min: int
    ngram_max: int


@dataclass(frozen=True, slots=True)
class BalanceConfig:
    length_bins: tuple[LengthBin, ...]
    mode: str
    seed: int

    def bin_of(self, words: int) -> LengthBin | None:
        """Faixa que contém `words` palavras, ou `None` se estiver fora de todas."""
        return next((b for b in self.length_bins if b.contains(words)), None)


@dataclass(frozen=True, slots=True)
class SplitConfig:
    ratios: tuple[float, float, float]
    seed: int
    min_stratum: int


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    short_band: tuple[int, int]
    long_band: tuple[int, int]
    batch_size: int
    temperature: float
    verify_labels: bool
    sleep_seconds: float


@dataclass(frozen=True, slots=True)
class SourcesConfig:
    #: Origens incluídas no build ("legacy", "mitigation").
    include_origins: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ControlConfig:
    """Opções da condição-controle (vazias no corpus principal).

    - `exclude_test_of`: diretório de splits de outro corpus; os textos do `test.jsonl` dele (e as
      quase-duplicatas) saem deste corpus antes do balanceamento, para que os dois modelos sejam
      comparados no mesmo teste sem vazamento.
    - `match_size_of`: manifesto de outro corpus; este corpus fica com o mesmo número por classe, para
      que a comparação isole a composição (comprimento) e não o tamanho do treino.
    """

    exclude_test_of: Path | None
    match_size_of: Path | None


@dataclass(frozen=True, slots=True)
class CorpusConfig:
    paths: CorpusPaths
    filters: FilterConfig
    near_dedup: NearDedupConfig
    balance: BalanceConfig
    split: SplitConfig
    generation: GenerationConfig
    sources: SourcesConfig
    control: ControlConfig
    source_file: Path


def _validate_bins(bins: tuple[LengthBin, ...], min_words: int) -> None:
    if not bins:
        raise ConfigError("balance.length_bins vazio")
    if bins[0].low != min_words:
        raise ConfigError(f"a primeira faixa deve começar em filters.min_words ({min_words})")
    for previous, current in zip(bins, bins[1:], strict=False):
        if current.low != previous.high + 1:
            raise ConfigError(f"faixas devem ser contíguas: {previous.name} seguida de {current.name}")
    for b in bins:
        if b.low > b.high:
            raise ConfigError(f"faixa inválida {b.low}-{b.high}")


def load_corpus_config(path: Path | None = None) -> CorpusConfig:
    """Lê e valida a configuração do corpus."""
    path = path or DEFAULT_CONFIG_PATH
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    try:
        p, f, n, b, s, g = (raw[k] for k in ("paths", "filters", "near_dedup", "balance", "split", "generation"))
        sources = SourcesConfig(include_origins=tuple(raw.get("sources", {}).get("include_origins", ORIGINS)))
        c = raw.get("control", {})
        control = ControlConfig(
            exclude_test_of=resolve_path(c["exclude_test_of"]) if c.get("exclude_test_of") else None,
            match_size_of=resolve_path(c["match_size_of"]) if c.get("match_size_of") else None,
        )
        # Sem `paths.reports`: corpus.toml → reports/corpus/principal; corpus_<nome>.toml → reports/corpus/<nome>.
        p = {"reports": f"reports/corpus/{path.stem.removeprefix('corpus').strip('_') or 'principal'}", **p}
        paths = CorpusPaths(**{key: resolve_path(value) for key, value in p.items()})
        filters = FilterConfig(**f)
        near = NearDedupConfig(**n)
        balance = BalanceConfig(
            length_bins=tuple(LengthBin(int(lo), int(hi)) for lo, hi in b["length_bins"]),
            mode=str(b["mode"]),
            seed=int(b["seed"]),
        )
        split = SplitConfig(ratios=tuple(float(x) for x in s["ratios"]), seed=int(s["seed"]),
                            min_stratum=int(s["min_stratum"]))
        generation = GenerationConfig(
            short_band=tuple(int(x) for x in g["short_band"]),
            long_band=tuple(int(x) for x in g["long_band"]),
            batch_size=int(g["batch_size"]),
            temperature=float(g["temperature"]),
            verify_labels=bool(g["verify_labels"]),
            sleep_seconds=float(g["sleep_seconds"]),
        )
    except (KeyError, TypeError) as exc:
        raise ConfigError(f"{path}: chave ausente ou inválida: {exc}") from exc

    if not 0.0 < near.threshold < 1.0:
        raise ConfigError("near_dedup.threshold deve estar em (0, 1)")
    if len(split.ratios) != 3 or abs(sum(split.ratios) - 1.0) > 1e-9:
        raise ConfigError(f"split.ratios deve ter 3 valores somando 1.0: {split.ratios}")
    for band in (generation.short_band, generation.long_band):
        if len(band) != 2 or band[0] > band[1] or band[0] < filters.min_words:
            raise ConfigError(f"faixa de geração inválida: {band}")
    if balance.mode not in {"global", "faixa", "faixa_fonte", "faixa_fonte_origem"}:
        raise ConfigError(f"balance.mode inválido: {balance.mode!r}")
    if unknown := set(sources.include_origins) - set(ORIGINS):
        raise ConfigError(f"sources.include_origins com origens desconhecidas: {sorted(unknown)}")
    _validate_bins(balance.length_bins, filters.min_words)
    if balance.length_bins[-1].high != filters.max_words:
        raise ConfigError("a última faixa deve terminar em filters.max_words")
    return CorpusConfig(paths, filters, near, balance, split, generation, sources, control, path)
