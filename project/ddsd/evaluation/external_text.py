"""Validação externa em texto: taxa de falso positivo em transcrições de fala espontânea real.

Fontes (partições de teste, só a coluna de texto, sem decodificar áudio):
- CORAA v1.1 (`gabrielrstan/CORAA-v1.1`): `metadata_test_final.csv`, estilo "Spontaneous Speech";
- NURC-SP (`nilc-nlp/CORAA-NURC-SP-Audio-Corpus`): `filtered/audios_test_metadata.csv`;
- TAGARELA (`freds0/TAGARELA`): `data/test-00000-of-00001.parquet`, coluna `sentence`,
  com os parágrafos quebrados em sentenças.

São conversas entre pessoas, sem anotação DD/NDD. O rótulo NDD é suposição de
domínio, e a métrica é a fração classificada como DD (taxa de falso positivo:
o assistente "acordaria à toa"). Para não medir memorização, saem as frases
iguais ou quase iguais (cosseno TF-IDF > 0,95) a qualquer frase do corpus de treino.
"""

from __future__ import annotations

import io
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from ddsd.common.io import read_jsonl, write_jsonl
from ddsd.corpus.config import CorpusConfig
from ddsd.corpus.near_dedup import find_near_duplicates, tfidf_matrix
from ddsd.observability.logger import get_logger
from ddsd.text.normalization import normalize_text, word_count

log = get_logger("avaliacao.externa")

HF = "https://huggingface.co/datasets"
SOURCES = {
    "coraa": f"{HF}/gabrielrstan/CORAA-v1.1/resolve/main/metadata_test_final.csv",
    "nurcsp": f"{HF}/nilc-nlp/CORAA-NURC-SP-Audio-Corpus/resolve/main/filtered/audios_test_metadata.csv",
    "tagarela": f"{HF}/freds0/TAGARELA/resolve/main/data/test-00000-of-00001.parquet",
}
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _download(url: str) -> bytes:
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    return response.content


def fetch_texts(corpus: str) -> list[str]:
    """Todas as transcrições candidatas do teste do corpus (texto original)."""
    if corpus == "coraa":
        frame = pd.read_csv(io.BytesIO(_download(SOURCES[corpus])))
        frame = frame[frame["speech_style"] == "Spontaneous Speech"]
        return frame["text"].dropna().astype(str).tolist()
    if corpus == "nurcsp":
        frame = pd.read_csv(io.BytesIO(_download(SOURCES[corpus])))
        return frame["text"].dropna().astype(str).tolist()
    if corpus == "tagarela":
        import pyarrow.parquet as pq

        table = pq.read_table(io.BytesIO(_download(SOURCES[corpus])), columns=["sentence"])
        texts: list[str] = []
        for paragraph in table.column("sentence").to_pylist():
            if paragraph:
                texts += [piece.strip() for piece in _SENTENCE_SPLIT.split(paragraph.strip()) if piece.strip()]
        return texts
    raise ValueError(corpus)


def prepare(cfg: CorpusConfig, out_dir: Path, sample_size: int = 1000, seed: int = 42) -> dict[str, Any]:
    """Baixa, normaliza, filtra, remove sobreposição com o corpus e sorteia `sample_size` por corpus."""
    train_texts = [row["text"] for split in ("train", "val", "test")
                   for row in read_jsonl(cfg.paths.frozen / "splits" / f"{split}.jsonl")]
    stats: dict[str, Any] = {}
    for corpus in SOURCES:
        raw = fetch_texts(corpus)
        seen: set[str] = set()
        candidates: list[tuple[str, str]] = []
        for text in raw:
            norm = normalize_text(text)
            if len(norm) <= cfg.filters.min_chars or word_count(norm) < cfg.filters.min_words or not norm.isascii():
                continue
            if norm in seen:
                continue
            seen.add(norm)
            candidates.append((text, norm))
        # Remove o que for igual ou quase igual a frases do corpus (memorização não conta).
        matrix = tfidf_matrix([n for _, n in candidates] + train_texts, (1, 2))
        pairs = find_near_duplicates(matrix, cfg.near_dedup.threshold)
        k = len(candidates)
        overlapping = {int(a) for a, b in zip(pairs.left, pairs.right, strict=True) if a < k <= b}
        kept = [c for i, c in enumerate(candidates) if i not in overlapping]
        rng = np.random.default_rng(seed)
        chosen = [kept[i] for i in sorted(rng.choice(len(kept), size=min(sample_size, len(kept)), replace=False))]
        rows = [{"corpus": corpus, "text_raw": t, "text": n, "words": word_count(n), "label": "NDD"} for t, n in chosen]
        write_jsonl(out_dir / f"{corpus}.jsonl", rows)
        stats[corpus] = {"brutas": len(raw), "candidatas_unicas": k, "sobrepostas_ao_corpus": len(overlapping),
                         "amostradas": len(rows),
                         "mediana_palavras": float(np.median([r["words"] for r in rows])) if rows else 0}
        log.info("%s: %d brutas, %d únicas, %d sobrepostas ao corpus, %d amostradas", corpus, len(raw), k,
                 len(overlapping), len(rows))
    return stats


def length_bin(words: int, bins: list[tuple[int, int, str]]) -> str:
    """Faixa de comprimento da frase. Fora das faixas do corpus vira "<mín" ou ">máx", reportada à parte."""
    for low, high, name in bins:
        if low <= words <= high:
            return name
    return f"<{bins[0][0]}" if words < bins[0][0] else f">{bins[-1][1]}"


def false_positive_report(predictions: dict[str, list[tuple[dict[str, Any], str, float]]],
                          bins: list[tuple[int, int, str]]) -> dict[str, Any]:
    """Taxa de FP por modelo × corpus, geral e por faixa de comprimento.

    Toda frase entra na taxa do corpus; as que ficam fora das faixas do corpus de treino (ex.: mais de
    50 palavras) aparecem numa faixa própria (">50"), em vez de serem descartadas.
    """
    names = [f"<{bins[0][0]}", *(n for _, _, n in bins), f">{bins[-1][1]}"]
    report: dict[str, Any] = {}
    for model, rows in predictions.items():
        per_corpus: dict[str, Any] = {}
        for corpus in sorted({r["corpus"] for r, _, _ in rows}):
            subset = [(r, label) for r, label, _ in rows if r["corpus"] == corpus]
            by_bin = Counter()
            dd_by_bin = Counter()
            for r, label in subset:
                name = length_bin(r["words"], bins)
                by_bin[name] += 1
                dd_by_bin[name] += label == "DD"
            per_corpus[corpus] = {
                "n": len(subset), "fp_rate": sum(label == "DD" for _, label in subset) / max(len(subset), 1),
                "por_faixa": {n: {"n": by_bin[n], "fp_rate": dd_by_bin[n] / by_bin[n]} for n in names if by_bin[n]},
            }
        report[model] = per_corpus
    return report


def paired_fp_bootstrap(labels_a: list[str], labels_b: list[str], n_bootstrap: int = 2000,
                        seed: int = 42, groups: list | None = None) -> dict[str, float]:
    """Diferença de taxa de FP (A - B) nas mesmas frases, por bootstrap pareado.

    Devolve as duas taxas, a diferença, o IC 95% por percentis e P(A tem menos FP). Com `groups`, sorteia as
    frases em vez dos itens (`ddsd.evaluation.resampling`).
    """
    from ddsd.evaluation.resampling import make_resampler

    a = np.asarray([label == "DD" for label in labels_a], dtype=float)
    b = np.asarray([label == "DD" for label in labels_b], dtype=float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError(f"predições desalinhadas ou vazias: {a.shape} × {b.shape}")
    rng = np.random.default_rng(seed)
    draw = make_resampler(a.size, groups)
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):  # um vetor de índices por vez: memória O(n)
        idx = draw(rng)
        diffs[i] = a[idx].mean() - b[idx].mean()
    return {"fp_a": float(a.mean()), "fp_b": float(b.mean()), "diferenca": float(a.mean() - b.mean()),
            "ic95_inf": float(np.percentile(diffs, 2.5)), "ic95_sup": float(np.percentile(diffs, 97.5)),
            "p_a_menos_fp": float(np.mean(diffs < 0))}
