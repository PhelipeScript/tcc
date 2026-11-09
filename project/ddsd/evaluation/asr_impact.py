"""Impacto do ASR no classificador, com TTS e com áudio real.

Para cada item de áudio com texto de referência:
1. carrega o áudio e reamostra para 16 kHz, como o pipeline;
2. transcreve com a mesma `WhisperBackend` e configuração do pipeline;
3. calcula o WER (jiwer, `normalize_text` nos dois lados) e a confiabilidade fuzzy;
4. classifica o texto de referência (limite superior) e a transcrição com cada modelo.

Saídas: métricas (referência × ASR nos mesmos itens), acurácia por faixa de WER,
Spearman entre a confiabilidade fuzzy e o WER, e um registro por item
(`registros.jsonl`) para análises posteriores.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jiwer
import numpy as np
import soundfile as sf

from ddsd.audio.resampler import StreamResampler
from ddsd.common.io import read_jsonl, write_jsonl
from ddsd.fuzzy.reliability import reliability_of
from ddsd.observability.logger import get_logger
from ddsd.realtime.config import MODEL_RATE
from ddsd.text.normalization import normalize_text

log = get_logger("avaliacao.asr")

TTS_VOICES = ("pt-BR-FranciscaNeural", "pt-BR-AntonioNeural", "pt-BR-ThalitaMultilingualNeural")
WER_BANDS = ((0.0, 0.0, "0"), (1e-9, 0.1, "(0;0,1]"), (0.1 + 1e-9, 0.3, "(0,1;0,3]"), (0.3 + 1e-9, 99.0, ">0,3"))
#: Pausas preenchidas, já normalizadas, das transcrições de referência do CORAA e do NURC-SP. O Whisper costuma
#: omiti-las, o que infla o WER sem mudar o sentido; o WER secundário as remove dos dois lados. "ha" fica fora:
#: depois da normalização, ele colide com o verbo "há".
FILLED_PAUSES = frozenset({"eh", "ah", "ahn", "ahm", "ehm", "uh", "uhm", "hum", "hm"})


@dataclass(frozen=True)
class AudioItem:
    item_id: str
    path: Path
    reference: str
    label: str
    group: str  # ex.: "tts", "quieto_interno", "coraa"


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return float(jiwer.wer(ref, hyp))


def without_filled_pauses(text: str) -> str:
    """Texto normalizado sem as pausas preenchidas (`FILLED_PAUSES`)."""
    return " ".join(word for word in normalize_text(text).split() if word not in FILLED_PAUSES)


def wer_without_filled_pauses(reference: str, hypothesis: str) -> float:
    """WER secundário: as pausas preenchidas saem da referência e da transcrição antes da comparação."""
    ref, hyp = without_filled_pauses(reference), without_filled_pauses(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return float(jiwer.wer(ref, hyp))


def load_16k(path: Path) -> np.ndarray:
    """Áudio mono a 16 kHz, com o mesmo reamostrador do frontend."""
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1).astype(np.float32)
    to16 = StreamResampler(rate, MODEL_RATE)
    return np.concatenate([to16.process(audio), to16.process(np.zeros(0, np.float32), last=True)])


def tts_items(test_split: Path, out_dir: Path, n: int = 750, seed: int = 42) -> list[AudioItem]:
    """Amostra estratificada (classe × faixa) do teste, sintetizada com vozes pt-BR do edge-tts (com cache)."""
    import edge_tts

    rows = list(read_jsonl(test_split))
    rng = random.Random(seed)
    strata: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        strata.setdefault(f"{row['label']}|{row['bin']}", []).append(row)
    per = max(1, n // len(strata))
    chosen = [r for key in sorted(strata) for r in rng.sample(strata[key], min(per, len(strata[key])))]
    rng.shuffle(chosen)
    chosen = chosen[:n]
    out_dir.mkdir(parents=True, exist_ok=True)

    async def synth(text: str, voice: str, path: Path) -> None:
        await edge_tts.Communicate(text, voice).save(str(path))

    items = []
    for i, row in enumerate(chosen):
        voice = TTS_VOICES[i % len(TTS_VOICES)]
        path = out_dir / tts_cache_name(row["text_raw"], voice)
        if not path.exists():
            partial = path.with_name(path.name + ".parcial")
            asyncio.run(synth(row["text_raw"], voice, partial))
            partial.rename(path)  # só um arquivo completo vira cache
        items.append(AudioItem(f"tts_{i:04d}", path, row["text_raw"], row["label"], f"tts:{voice}"))
    write_jsonl(out_dir / "itens.jsonl", ({"id": it.item_id, "arquivo": it.path.name, "texto": it.reference,
                                          "rotulo": it.label, "origem": it.group} for it in items))
    return items


def tts_cache_name(text: str, voice: str) -> str:
    """Nome do áudio em cache derivado da voz e do texto: mudar a amostra nunca reaproveita o áudio errado."""
    return f"tts_{hashlib.sha1(f'{voice}\n{text}'.encode()).hexdigest()[:16]}.mp3"


def recording_items(manifest: Path, base: Path) -> list[AudioItem]:
    """Gravações do autor (`ddsd record-eval-set`); fala livre sem referência entra só no FP."""
    return [AudioItem(f"{r['scenario']}/{r['item']}", base / r["file"], r["reference"], r["label"], r["scenario"])
            for r in read_jsonl(manifest) if r["label"] in ("DD", "NDD")]


def transcribe_items(items: list[AudioItem], backend: Any) -> list[dict[str, Any]]:
    """Transcreve com a mesma configuração do pipeline e calcula WER e confiabilidade."""
    records = []
    for k, item in enumerate(items, start=1):
        audio = load_16k(item.path)
        result = backend.transcribe(audio)
        reliability = reliability_of(result.segments)
        records.append({
            "id": item.item_id, "group": item.group, "label": item.label,
            "reference": item.reference, "hypothesis_raw": result.text_raw,
            "hypothesis": normalize_text(result.text_raw), "reference_norm": normalize_text(item.reference),
            "wer": wer(item.reference, result.text_raw) if item.reference else None,
            "wer_sem_pausas": wer_without_filled_pauses(item.reference, result.text_raw) if item.reference else None,
            "reliability": reliability.score, "reliability_label": reliability.label,
            "asr_latency_s": result.latency_s, "audio_s": result.audio_s,
        })
        if k % 50 == 0:
            log.info("transcritos %d/%d", k, len(items))
    return records


def classify_records(records: list[dict[str, Any]], classifiers: dict[str, Any]) -> None:
    """Acrescenta a predição de cada modelo no texto de referência e na transcrição."""
    for name, clf in classifiers.items():
        for field in ("reference_norm", "hypothesis"):
            texts = [r[field] or "" for r in records]
            nonempty = [i for i, t in enumerate(texts) if t]
            preds = clf.predict_batch([texts[i] for i in nonempty]) if nonempty else []
            for i, p in zip(nonempty, preds, strict=True):
                records[i][f"{name}:{field}"] = p.label
                records[i][f"{name}:{field}:p"] = p.p_dd
            for i in set(range(len(texts))) - set(nonempty):
                records[i][f"{name}:{field}"] = "NDD"  # sem texto: o sistema não ativa


def summarize(records: list[dict[str, Any]], models: list[str]) -> dict[str, Any]:
    """Acurácia, FP e FN por modelo × condição; acurácia por faixa de WER; Spearman(r, WER).

    `referencia` e `asr` usam os mesmos itens (os que têm texto de referência), então a diferença
    entre os dois é a queda causada pelo ASR. A fala livre, sem roteiro, não tem referência e fica à
    parte, em `asr_sem_referencia` (na prática, a taxa de FP dela).
    """
    from scipy.stats import spearmanr

    def rates(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
        labeled = [r for r in rows if r["label"] in ("DD", "NDD")]
        dd = [r for r in labeled if r["label"] == "DD"]
        ndd = [r for r in labeled if r["label"] == "NDD"]
        return {"n": len(labeled),
                "acuracia": float(np.mean([r[key] == r["label"] for r in labeled])) if labeled else float("nan"),
                "fp_rate": float(np.mean([r[key] == "DD" for r in ndd])) if ndd else float("nan"),
                "fn_rate": float(np.mean([r[key] == "NDD" for r in dd])) if dd else float("nan")}

    with_ref = [r for r in records if r["wer"] is not None]
    without_ref = [r for r in records if r["wer"] is None]
    summary: dict[str, Any] = {"n": len(records), "n_com_referencia": len(with_ref),
                               "wer_medio": float(np.mean([r["wer"] for r in with_ref])) if with_ref else None}
    if with_ref and all(r.get("wer_sem_pausas") is not None for r in with_ref):
        summary["wer_medio_sem_pausas"] = float(np.mean([r["wer_sem_pausas"] for r in with_ref]))
    if len(with_ref) > 2:
        rho, p = spearmanr([r["reliability"] for r in with_ref], [r["wer"] for r in with_ref])
        summary["spearman_confiabilidade_wer"] = {"rho": float(rho), "p": float(p)}
    for model in models:
        summary[model] = {"referencia": rates(with_ref, f"{model}:reference_norm"),
                          "asr": rates(with_ref, f"{model}:hypothesis")}
        if without_ref:
            summary[model]["asr_sem_referencia"] = rates(without_ref, f"{model}:hypothesis")
        bands = {}
        for lo, hi, name in WER_BANDS:
            sel = [r for r in with_ref if lo <= r["wer"] <= hi]
            if sel:
                bands[name] = {"n": len(sel), **rates(sel, f"{model}:hypothesis")}
        summary[model]["por_faixa_wer"] = bands
    return summary


def scored_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Itens DD/NDD com texto de referência, na ordem dos registros: os que entram na hipótese."""
    return [r for r in records if r["wer"] is not None and r["label"] in ("DD", "NDD")]


def text_predictions(records: list[dict[str, Any]], models: list[str], field: str = "hypothesis"
                     ) -> dict[str, dict[str, np.ndarray]]:
    """Predições de cada modelo na transcrição (`hypothesis`) ou no texto de referência (`reference_norm`).

    Sai no formato do bootstrap pareado do teste interno (`ddsd.evaluation.bootstrap`). Só entram itens com
    texto de referência. Sem texto, o sistema não ativa: rótulo NDD e p(DD) = 0.
    """
    from ddsd.training.config import LABEL2ID

    rows = scored_rows(records)
    return {m: {"text": np.array([r[field] or "" for r in rows], dtype=object),
                "y_true": np.array([LABEL2ID[r["label"]] for r in rows]),
                "y_pred": np.array([LABEL2ID[r[f"{m}:{field}"]] for r in rows]),
                "y_score": np.array([r.get(f"{m}:{field}:p", 0.0) for r in rows], dtype=float)}
            for m in models}


def drop_by_wer(records: list[dict[str, Any]], models: list[str]) -> dict[str, Any]:
    """Queda pareada do acerto (referência - transcrição) por faixa de WER, e Spearman(WER, queda) por item.

    A queda compara o mesmo item nas duas versões do texto, então separa o erro do ASR da dificuldade da
    frase (o acerto só na transcrição misturaria as duas coisas).
    """
    from scipy.stats import spearmanr

    rows = [r for r in records if r["wer"] is not None and r["label"] in ("DD", "NDD")]
    wer = np.array([r["wer"] for r in rows])
    out: dict[str, Any] = {"faixas": [name for *_, name in WER_BANDS],
                           "n_por_faixa": {name: int(((wer >= lo) & (wer <= hi)).sum()) for lo, hi, name in WER_BANDS}}
    for m in models:
        drop = np.array([float(r[f"{m}:reference_norm"] == r["label"]) - float(r[f"{m}:hypothesis"] == r["label"])
                         for r in rows])
        rho, p = spearmanr(wer, drop) if np.ptp(drop) > 0 else (float("nan"), float("nan"))
        out[m] = {"queda_media": float(drop.mean()), "spearman_wer_queda": {"rho": float(rho), "p": float(p)},
                  "queda_por_faixa": {name: float(drop[(wer >= lo) & (wer <= hi)].mean())
                                      for lo, hi, name in WER_BANDS if ((wer >= lo) & (wer <= hi)).any()}}
    return out


def paired_drop_bootstrap(records: list[dict[str, Any]], model_a: str, model_b: str, n_bootstrap: int = 2000,
                          seed: int = 42, group_field: str | None = None) -> dict[str, float]:
    """Queda de acerto causada pelo ASR (texto de referência → transcrição), A - B, nos mesmos itens.

    Por item, a queda de um modelo é acerto_na_referência - acerto_na_transcrição (-1, 0 ou 1). Devolve as
    quedas médias, a diferença com IC 95% por bootstrap pareado e P(A perde menos que B). Só entram itens
    com texto de referência (a fala livre não tem). Com `group_field`, sorteia as frases em vez dos itens.
    """
    from ddsd.evaluation.resampling import make_resampler

    items = [r for r in records if r["wer"] is not None]

    def drop(model: str) -> np.ndarray:
        return np.asarray([float(r[f"{model}:reference_norm"] == r["label"]) - float(r[f"{model}:hypothesis"] == r["label"])
                           for r in items])

    a, b = drop(model_a), drop(model_b)
    if a.size == 0:
        raise ValueError("nenhum item com texto de referência")
    rng = np.random.default_rng(seed)
    draw = make_resampler(a.size, None if group_field is None else [r[group_field] for r in items])
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = draw(rng)
        diffs[i] = a[idx].mean() - b[idx].mean()
    return {"n": int(a.size), "queda_a": float(a.mean()), "queda_b": float(b.mean()),
            "diferenca": float(a.mean() - b.mean()), "ic95_inf": float(np.percentile(diffs, 2.5)),
            "ic95_sup": float(np.percentile(diffs, 97.5)), "p_a_perde_menos": float(np.mean(diffs < 0))}


def save_records(path: Path, records: list[dict[str, Any]]) -> None:
    write_jsonl(path, records)
