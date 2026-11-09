"""Geração da mitigação do viés de comprimento, em pares.

O corpus inicial quase não tem NDD curto (2-6 palavras) nem DD longo (11-24).
A mitigação gera as duas classes, em quantidades iguais, em cada uma dessas
faixas, com o mesmo modelo e a mesma versão de prompt (um prompt por classe; cada
lote gera uma classe só). Junto com o balanceamento
`faixa_fonte_origem`, isso garante que, dentro de uma célula (faixa, fonte,
origem), nada além da semântica separa DD de NDD: nem comprimento, nem modelo
gerador, nem estilo de prompt.

Cada lote pede `n` frases de uma classe com exatamente `k` palavras. A resposta
é filtrada (wake-words, casa inteligente, faixa de comprimento, duplicata,
conflito com a outra classe) e anexada a
`data/corpus/sources/mitigation/<fonte>/{dd,ndd}.jsonl`, com a proveniência
completa de cada linha.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ddsd.common.io import append_jsonl
from ddsd.corpus.config import CorpusConfig, LengthBin
from ddsd.corpus.generation.prompts import (HARD_PROMPT_VERSION, PROMPT_VERSION, PromptSpec, render_hard_judge,
                                               render_hard_prompt, render_judge, render_prompt)
from ddsd.corpus.sources import load_sources
from ddsd.observability.logger import get_logger
from ddsd.text.normalization import normalize_text, word_count

log = get_logger("corpus.geracao")

#: Lotes seguidos sem nenhuma frase aceita até a célula ser considerada esgotada.
MAX_EMPTY_BATCHES = 8

WAKE_WORDS = ("alexa", "siri", "ok google", "hey google", "assistente")
SMART_HOME = (
    "ar condicionado", "aquecedor", "cortina", "lampada", "luz do quarto", "luz do ambiente",
    "luz da sala", "maquina de cafe", "cafeteira", "ventilador", "geladeira", "chaleira", "fogao",
    "acende a luz", "apaga a luz", "acender a luz", "aspirador", "camera de seguranca",
)


def parse_json_array(raw: str) -> list[str]:
    """Extrai o array JSON de strings da resposta (tolera cercas de markdown e objetos)."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1].removeprefix("json").strip()
    start, end = text.find("["), text.rfind("]")
    candidates = [text[start : end + 1]] if start != -1 and end > start else []
    candidates.append(text)
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, list)), None)
        if isinstance(value, list):
            return [str(v) for v in value if isinstance(v, (str, int, float))]
    raise ValueError(f"array JSON não encontrado: {raw[:200]!r}")


def screening_reason(normalized: str) -> str | None:
    """Motivo de rejeição por conteúdo (sobre o texto normalizado), ou `None`."""
    if not normalized:
        return "vazio"
    padded = f" {normalized} "
    if any(f" {w} " in padded for w in WAKE_WORDS):
        return "palavra_ativacao"
    if any(f" {w} " in padded for w in SMART_HOME):
        return "casa_inteligente"
    return None


@dataclass
class CorpusState:
    """Textos normalizados existentes (por classe) e contagens por (origem, fonte, faixa, classe)."""

    texts: dict[str, set[str]]
    counts: Counter
    bins: tuple[LengthBin, ...]

    @classmethod
    def load(cls, cfg: CorpusConfig) -> CorpusState:
        rows, _, _ = load_sources(cfg)
        texts: dict[str, set[str]] = {"DD": set(), "NDD": set()}
        counts: Counter = Counter()
        for row in rows:
            text = normalize_text(row.text_raw)
            if not text or text in texts[row.label]:
                continue
            texts[row.label].add(text)
            length_bin = cfg.balance.bin_of(word_count(text))
            if length_bin is not None:
                counts[(row.origin, row.source, length_bin.name, row.label)] += 1
        return cls(texts=texts, counts=counts, bins=cfg.balance.length_bins)

    def mitigation_count(self, source: str, bin_name: str, label: str) -> int:
        """Frases já aceitas da mitigação na célula."""
        return self.counts[("mitigation", source, bin_name, label)]


@dataclass
class Validation:
    accepted: list[dict[str, Any]] = field(default_factory=list)
    rejected: Counter = field(default_factory=Counter)


@dataclass(frozen=True, slots=True)
class Candidate:
    raw: str
    normalized: str
    words: int
    bin: str


def screen_candidates(candidates: Iterable[str], label: str, band: tuple[int, int],
                      state: CorpusState) -> tuple[list[Candidate], Counter]:
    """Filtros determinísticos (conteúdo, faixa, duplicata, conflito), sem alterar o estado."""
    kept: list[Candidate] = []
    rejected: Counter = Counter()
    other = "NDD" if label == "DD" else "DD"
    seen_in_batch: set[str] = set()
    for raw in candidates:
        raw = " ".join(str(raw).split())
        normalized = normalize_text(raw)
        reason = screening_reason(normalized)
        words = word_count(normalized)
        if reason is None and not band[0] <= words <= band[1]:
            reason = "fora_da_faixa"
        if reason is None and (normalized in state.texts[label] or normalized in seen_in_batch):
            reason = "duplicata"
        if reason is None and normalized in state.texts[other]:
            reason = "conflito_outra_classe"
        if reason is not None:
            rejected[reason] += 1
            continue
        seen_in_batch.add(normalized)
        kept.append(Candidate(raw, normalized, words, next(b.name for b in state.bins if b.contains(words))))
    return kept, rejected


def parse_judge_labels(raw: str, expected: int) -> list[str] | None:
    """Rótulos do juiz, ou `None` se a resposta não tiver exatamente `expected` rótulos válidos."""
    try:
        labels = [str(x).strip().upper() for x in parse_json_array(raw)]
    except ValueError:
        return None
    if len(labels) != expected or any(lab not in {"DD", "NDD"} for lab in labels):
        return None
    return labels


def accept(candidates: Iterable[Candidate], label: str, state: CorpusState,
           provenance: dict[str, Any]) -> list[dict[str, Any]]:
    """Registra os candidatos aprovados no estado e devolve as linhas a gravar."""
    rows = []
    for c in candidates:
        state.texts[label].add(c.normalized)
        rows.append({"text": c.raw, "label": label, "strategy": provenance.get("strategy", "comprimento"),
                     "words": c.words, "bin": c.bin, **{k: v for k, v in provenance.items() if k != "strategy"}})
    return rows


def validate_candidates(candidates: Iterable[str], label: str, band: tuple[int, int],
                        state: CorpusState, provenance: dict[str, Any]) -> Validation:
    """Filtros determinísticos + aceitação (sem juiz), usado na ingestão de arquivos."""
    kept, rejected = screen_candidates(candidates, label, band, state)
    return Validation(accepted=accept(kept, label, state, provenance), rejected=rejected)


class SourceLock:
    """Impede dois geradores escrevendo na mesma fonte (o JSONL corromperia)."""

    def __init__(self, directory: Path) -> None:
        self.path = directory / ".lock"

    def __enter__(self) -> SourceLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            pid = int(self.path.read_text() or 0)
            try:
                os.kill(pid, 0)
            except (ProcessLookupError, ValueError):
                log.warn("trava órfã de PID %s removida", pid)
            else:
                raise RuntimeError(f"outro gerador (PID {pid}) já escreve em {self.path.parent}")
        self.path.write_text(str(os.getpid()))
        return self

    def __exit__(self, *exc: object) -> None:
        self.path.unlink(missing_ok=True)


def _taxonomy_combos(cfg: CorpusConfig, seed: int) -> list[tuple[str, tuple[str, ...], str, str]]:
    taxonomy = yaml.safe_load(cfg.paths.taxonomy.read_text(encoding="utf-8"))
    combos = [(cat["label"], tuple(cat["examples"]), ctx, tone)
              for cat in taxonomy["categories"].values()
              for ctx in taxonomy["contexts"] for tone in taxonomy["tones"]]
    random.Random(seed).shuffle(combos)
    return combos


def target_bins(cfg: CorpusConfig) -> list[LengthBin]:
    """Faixas-alvo da mitigação: as curtas e as longas (as duas classes em cada uma)."""
    short, long_ = cfg.generation.short_band, cfg.generation.long_band
    return [b for b in cfg.balance.length_bins
            if (b.low >= short[0] and b.high <= short[1]) or (b.low >= long_[0] and b.high <= long_[1])]


STRATEGIES = ("comprimento", "negativos_dificeis")


def strategy_bins(cfg: CorpusConfig, strategy: str) -> list[LengthBin]:
    """Faixas-alvo: curtas e longas em `comprimento`; só as curtas em `negativos_dificeis`."""
    if strategy == "negativos_dificeis":
        short = cfg.generation.short_band
        return [b for b in cfg.balance.length_bins if b.low >= short[0] and b.high <= short[1]]
    return target_bins(cfg)


def pair_targets(state: CorpusState, cfg: CorpusConfig, source: str, per_cell: int,
                 strategy: str = "comprimento") -> dict[tuple[str, str], int]:
    """Quanto falta, em cada (classe, faixa-alvo), para a fonte chegar a `per_cell` frases."""
    targets = strategy_bins(cfg, strategy)
    return {(label, b.name): max(0, per_cell - state.mitigation_count(source, b.name, label))
            for b in targets for label in ("DD", "NDD")}


def _verify(client: Any, screened: list[Candidate], label: str, rejected: Counter,
            judge: Any = render_judge) -> list[Candidate]:
    """Autoverificação: mantém só as frases que o modelo, como juiz, classifica com o rótulo pretendido."""
    from ddsd.corpus.generation.providers import QuotaExhausted

    try:
        verdict = parse_judge_labels(client.complete(judge([c.raw for c in screened]), 0.0), len(screened))
    except QuotaExhausted:
        raise
    except Exception as exc:  # noqa: BLE001 - sem veredito, o lote é descartado (não entra nada sem verificar)
        log.warn("autoverificação falhou: %s", exc)
        verdict = None
    if verdict is None:
        rejected["juiz_invalido"] += len(screened)
        return []
    kept = [c for c, v in zip(screened, verdict, strict=True) if v == label]
    rejected["juiz_discorda"] += len(screened) - len(kept)
    return kept


def run_generation(cfg: CorpusConfig, provider: str, *, per_cell: int, source_name: str | None = None,
                   sleep: float | None = None, seed: int | None = None, batch_size: int | None = None,
                   strategy: str = "comprimento") -> Counter:
    """Gera, para a fonte, até `per_cell` frases por (classe, faixa-alvo). Devolve as aceitas por célula."""
    from ddsd.corpus.generation.providers import PROVIDERS, LLMClient, QuotaExhausted

    if strategy not in STRATEGIES:
        raise ValueError(f"estratégia desconhecida: {strategy!r}")
    client = LLMClient(PROVIDERS[provider])
    hard = strategy == "negativos_dificeis"
    # Cada estratégia tem fonte própria: DD e NDD de uma célula vêm do mesmo modelo e da mesma versão de prompt.
    source = source_name or (f"{provider}_negdif" if hard else provider)
    sleep = cfg.generation.sleep_seconds if sleep is None else sleep
    rng = random.Random(seed if seed is not None else time.time_ns())
    combos = _taxonomy_combos(cfg, rng.randrange(1 << 30))
    out_dir = cfg.paths.sources / "mitigation" / source
    produced: Counter = Counter()
    rejected: Counter = Counter()
    bin_by_name = {b.name: b for b in cfg.balance.length_bins}

    with SourceLock(out_dir):
        state = CorpusState.load(cfg)
        remaining = pair_targets(state, cfg, source, per_cell, strategy)
        log.info("fonte %s (modelo %s): faltam %d frases para %d por célula", source, client.model,
                 sum(remaining.values()), per_cell)
        batch = 0
        failures = 0
        stalls: Counter = Counter()
        while any(v > 0 for v in remaining.values()):
            # A célula mais atrasada primeiro mantém DD e NDD equilibrados durante a execução.
            label, bin_name = max(remaining, key=lambda k: (remaining[k], rng.random()))
            length_bin = bin_by_name[bin_name]
            words = rng.randint(length_bin.low, min(length_bin.high, cfg.generation.long_band[1]))
            category, examples, context, tone = combos[batch % len(combos)]
            spec = PromptSpec(label=label, words=words, n=batch_size or cfg.generation.batch_size,
                              category=category, examples=examples, context=context, tone=tone)
            band = cfg.generation.short_band if words <= cfg.generation.short_band[1] else cfg.generation.long_band
            batch += 1
            prompt = render_hard_prompt(spec, rng) if hard else render_prompt(spec, rng, cfg.generation.short_band[1])
            try:
                candidates = parse_json_array(client.complete(prompt, cfg.generation.temperature))
            except QuotaExhausted as exc:
                log.warn("%s - geração interrompida; rode o mesmo comando depois para continuar", exc)
                break
            except Exception as exc:  # noqa: BLE001 - falha de um lote não interrompe a geração
                failures += 1
                log.warn("lote %d falhou (%s|%d palavras): %s", batch, label, words, exc)
                if failures >= 20:
                    log.error("20 falhas seguidas; interrompendo (verifique chave, cota e modelo)")
                    break
                time.sleep(sleep)
                continue
            failures = 0
            provenance = {"source": source, "model": client.model, "provider": provider, "strategy": strategy,
                          "prompt_version": HARD_PROMPT_VERSION if hard else PROMPT_VERSION,
                          "target_words": words, "category": category, "context": context, "tone": tone,
                          "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            screened, batch_rejected = screen_candidates(candidates, label, band, state)
            if screened and cfg.generation.verify_labels:
                try:
                    screened = _verify(client, screened, label, batch_rejected,
                                       render_hard_judge if hard else render_judge)
                except QuotaExhausted as exc:
                    log.warn("%s - lote descartado sem verificação; geração interrompida", exc)
                    break
                provenance["verified_by"] = client.model
            # Só entra o que ainda tem vaga na célula, para manter DD e NDD pareados.
            fitting = []
            for c in screened:
                key = (label, c.bin)
                if remaining.get(key, 0) > 0:
                    remaining[key] -= 1
                    produced[key] += 1
                    fitting.append(c)
                else:
                    batch_rejected["celula_cheia"] += 1
            rejected.update(batch_rejected)
            # Célula que não rende frase nova em vários lotes seguidos (faixas de 2-3 palavras
            # esgotam o vocabulário) é encerrada, para não travar a geração.
            stalls[(label, bin_name)] = 0 if fitting else stalls[(label, bin_name)] + 1
            if stalls[(label, bin_name)] >= MAX_EMPTY_BATCHES and remaining[(label, bin_name)] > 0:
                log.warn("célula %s|%s esgotada após %d lotes vazios (faltavam %d)", label, bin_name,
                         MAX_EMPTY_BATCHES, remaining[(label, bin_name)])
                remaining[(label, bin_name)] = 0
            accepted = accept(fitting, label, state, provenance)
            if accepted:
                append_jsonl(out_dir / f"{label.lower()}.jsonl", accepted)
            log.info("lote %d %s|%dp: +%d (faltam %d) rejeitadas %s", batch, label, words, len(accepted),
                     sum(remaining.values()), dict(batch_rejected))
            time.sleep(sleep)
    log.info("fim: %d frases aceitas %s | rejeições %s", sum(produced.values()),
             {f"{k[0]}|{k[1]}": v for k, v in sorted(produced.items())}, dict(rejected))
    return produced


def ingest_file(cfg: CorpusConfig, source: str, label: str, path: Path,
                provenance: dict[str, Any]) -> Validation:
    """Valida e anexa frases geradas fora dos provedores HTTP (ex.: subagentes Claude).

    O arquivo pode ser JSONL (`{"text": ...}` por linha) ou ter um array JSON por linha.
    """
    candidates: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        value = json.loads(line)
        if isinstance(value, list):
            candidates.extend(str(v) for v in value)
        elif isinstance(value, dict) and "text" in value:
            candidates.append(str(value["text"]))
    targets = target_bins(cfg)
    band_lo, band_hi = min(b.low for b in targets), max(b.high for b in targets)
    out_dir = cfg.paths.sources / "mitigation" / source
    with SourceLock(out_dir):
        state = CorpusState.load(cfg)
        # Aceita qualquer faixa curta ou longa; fora delas (7-10 palavras) não há déficit a cobrir.
        short = validate_candidates([c for c in candidates
                                     if word_count(normalize_text(c)) <= cfg.generation.short_band[1]],
                                    label, cfg.generation.short_band, state, provenance)
        long_ = validate_candidates([c for c in candidates
                                     if word_count(normalize_text(c)) > cfg.generation.short_band[1]],
                                    label, cfg.generation.long_band, state, provenance)
        result = Validation(accepted=short.accepted + long_.accepted, rejected=short.rejected + long_.rejected)
        if result.accepted:
            append_jsonl(out_dir / f"{label.lower()}.jsonl", result.accepted)
    log.info("ingestão %s/%s de %s: %d aceitas, rejeitadas %s (faixas %d-%d)", source, label, path.name,
             len(result.accepted), dict(result.rejected), band_lo, band_hi)
    return result
