"""Avaliação ponta a ponta: gravações reais passam pelo pipeline completo em ritmo de tempo real.

Cada arquivo é tocado no mesmo `AudioRingBuffer` do microfone (via `FileAudioSource`),
seguido de 1,5 s de silêncio. O pipeline faz 16 kHz → VAD → fala completa → Whisper →
fuzzy → classificadores → decisor. A saída por arquivo é o conjunto de decisões das
falas que começaram dentro dele. O sistema "ativou" se alguma dessas falas foi DD.

Métricas por grupo (cenário/corpus), com o classificador ativo da configuração.
- DD: taxa de ativação (recall do sistema);
- NDD (frases do roteiro e conversa real dos corpora): taxa de ativação (FP);
- NDD_LIVRE (fala livre de ~60 s, sem roteiro), à parte: um arquivo tem várias falas, então a
  ativação por arquivo não é comparável à das frases. Entram também o FP por fala e as ativações
  por minuto;
- RUÍDO (só ambiente): falas detectadas e ativações (falsos alarmes);
- latência do fim da fala até a decisão (p50/p95);
- política offline "confiabilidade baixa → NDD": como mudariam FP e FN.
"""

from __future__ import annotations

import io
import random
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import soundfile as sf

from ddsd.audio.resampler import StreamResampler
from ddsd.common.io import read_jsonl, write_jsonl
from ddsd.evaluation.asr_impact import AudioItem
from ddsd.observability.logger import get_logger
from ddsd.realtime.config import CAPTURE_RATE

log = get_logger("avaliacao.pipeline")

#: Rótulo da fala livre das gravações (NDD sem roteiro), resumida à parte das frases NDD.
FREE_SPEECH = "NDD_LIVRE"
LABELS = ("DD", "NDD", FREE_SPEECH, "RUIDO")

PUBLIC_AUDIO = {
    "coraa": ("https://huggingface.co/datasets/gabrielrstan/CORAA-v1.1/resolve/main/test.zip",
              "https://huggingface.co/datasets/gabrielrstan/CORAA-v1.1/resolve/main/metadata_test_final.csv"),
    "nurcsp": ("https://huggingface.co/datasets/nilc-nlp/CORAA-NURC-SP-Audio-Corpus/resolve/main/test.tar.gz",
               "https://huggingface.co/datasets/nilc-nlp/CORAA-NURC-SP-Audio-Corpus/resolve/main/filtered/audios_test_metadata.csv"),
}


def _download(url: str, path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=600) as response:
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 22):
                handle.write(chunk)
    tmp.rename(path)
    return path


def prepare_public_audio(out_dir: Path, per_corpus: int = 300, seed: int = 42) -> list[dict[str, Any]]:
    """Sorteia falas espontâneas de 1,5-15 s do teste do CORAA e do NURC-SP e extrai só esses áudios."""
    rng = random.Random(seed)
    manifest: list[dict[str, Any]] = []
    for corpus, (audio_url, meta_url) in PUBLIC_AUDIO.items():
        meta = pd.read_csv(io.BytesIO(requests.get(meta_url, timeout=300).content))
        if corpus == "coraa":
            meta = meta[meta["speech_style"] == "Spontaneous Speech"]
            meta = meta[(meta["up_votes"] > meta["down_votes"]) & (meta["votes_for_second_voice"] == 0)]
        else:
            meta = meta[(meta["quality"] == "high") & meta["duration"].between(1.5, 15)]
        rows = meta.dropna(subset=["text"]).to_dict("records")
        chosen = rng.sample(rows, min(per_corpus * 2, len(rows)))  # sobra para descartar fora da duração
        archive = _download(audio_url, out_dir / "download" / Path(audio_url).name)
        wanted = {r["file_path"]: r for r in chosen}
        extracted = 0
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as zf:
                names = {n.split("/", 1)[-1] if not n.startswith("test/") else n: n for n in zf.namelist()}
                for key, row in wanted.items():
                    member = names.get(key) or names.get(key.split("/", 1)[-1])
                    if member is None:
                        continue
                    data, rate = sf.read(io.BytesIO(zf.read(member)), dtype="float32")
                    extracted += _keep(out_dir, corpus, row, data, rate, manifest, per_corpus, extracted)
                    if extracted >= per_corpus:
                        break
        else:
            with tarfile.open(archive, "r|gz") as tf:
                for member in tf:
                    key = member.name.split("/", 1)[-1] if member.name not in wanted else member.name
                    row = wanted.get(member.name) or wanted.get(f"test/{key}")
                    if row is None or not member.isfile():
                        continue
                    data, rate = sf.read(io.BytesIO(tf.extractfile(member).read()), dtype="float32")
                    extracted += _keep(out_dir, corpus, row, data, rate, manifest, per_corpus, extracted)
                    if extracted >= per_corpus:
                        break
        log.info("%s: %d áudios extraídos", corpus, extracted)
    write_jsonl(out_dir / "manifest.jsonl", manifest)
    return manifest


def _keep(out_dir: Path, corpus: str, row: dict[str, Any], data: np.ndarray, rate: int,
          manifest: list[dict[str, Any]], limit: int, extracted: int) -> int:
    if data.ndim > 1:
        data = data.mean(axis=1)
    duration = len(data) / rate
    if extracted >= limit or not 1.5 <= duration <= 15.0:
        return 0
    path = out_dir / corpus / f"{corpus}_{extracted:04d}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, rate, subtype="PCM_16")
    manifest.append({"id": path.stem, "file": str(path.relative_to(out_dir)), "corpus": corpus, "label": "NDD",
                     "reference": str(row["text"]), "duration_s": round(duration, 2), "origin": row["file_path"]})
    return 1


def public_items(out_dir: Path) -> list[AudioItem]:
    return [AudioItem(r["id"], out_dir / r["file"], r["reference"], r["label"], r["corpus"])
            for r in read_jsonl(out_dir / "manifest.jsonl")]


def recording_items(manifest: Path, base: Path) -> list[AudioItem]:
    """Gravações do autor: frases, roteiro de conversa, fala livre (NDD sem roteiro, à parte) e ambiente."""
    return [AudioItem(f"{r['scenario']}/{r['item']}", base / r["file"], r["reference"],
                      FREE_SPEECH if r["item"] == "fala_livre" else r["label"], r["scenario"])
            for r in read_jsonl(manifest)]


def tts_items(tts_dir: Path) -> list[AudioItem]:
    """As frases do teste em voz sintética, as mesmas do impacto do ASR (`data/eval/tts/itens.jsonl`)."""
    return [AudioItem(r["id"], tts_dir / r["arquivo"], r["texto"], r["rotulo"], r["origem"])
            for r in read_jsonl(tts_dir / "itens.jsonl")]


def to_capture_rate(path: Path) -> np.ndarray:
    """Áudio mono na taxa típica do microfone (48 kHz), para o arquivo entrar como uma captura."""
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1).astype(np.float32)
    resampler = StreamResampler(rate, CAPTURE_RATE)
    return np.concatenate([resampler.process(audio), resampler.process(np.zeros(0, np.float32), last=True)])


def run_items(config_path: Path | None, items: list[AudioItem], overrides: dict[str, Any] | None = None
              ) -> list[dict[str, Any]]:
    """Toca os itens no pipeline (tempo real) e devolve um registro por item com as decisões."""
    from ddsd.realtime.config import load_pipeline_config
    from ddsd.realtime.pipeline import Pipeline, utterances_for_file

    cfg = load_pipeline_config(config_path, **(overrides or {}))
    pipeline = Pipeline(cfg)
    pipeline.start(file_rate=CAPTURE_RATE)
    records = []
    try:
        pipeline.play(np.zeros(CAPTURE_RATE // 2, dtype=np.float32), trailing_silence_s=0.5)
        for k, item in enumerate(items, start=1):
            audio = to_capture_rate(item.path)
            start16, end16 = pipeline.play(audio, trailing_silence_s=1.5)
            pipeline.wait_idle(timeout_s=90)
            decisions = utterances_for_file(pipeline.decisions, start16, end16)
            records.append({
                "id": item.item_id, "group": item.group, "label": item.label, "reference": item.reference,
                "classifier": cfg.classifiers.active, "audio_s": len(audio) / CAPTURE_RATE,
                "utterances": len(decisions), "activated": any(d.label == "DD" for d in decisions),
                "activated_low_policy": any(d.label == "DD" and not d.reliability_alert for d in decisions),
                "decisions": [d.to_dict() for d in decisions],
                "latencies_s": [d.latency_s for d in decisions if d.latency_s is not None],
            })
            if k % 25 == 0:
                log.info("pipeline: %d/%d itens", k, len(items))
    finally:
        pipeline.stop()
    return records


def _per_utterance(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Fala livre: FP por fala detectada e ativações por minuto de áudio (com e sem a política offline)."""
    decisions = [d for r in rows for d in r["decisions"]]
    dd = sum(d["label"] == "DD" for d in decisions)
    dd_policy = sum(d["label"] == "DD" and not d["reliability"]["alert"] for d in decisions)
    minutes = sum(r["audio_s"] for r in rows) / 60
    nan = float("nan")
    return {"falas": len(decisions), "minutos": minutes,
            "fp_por_fala": dd / len(decisions) if decisions else nan,
            "fp_por_fala_politica_baixa_ndd": dd_policy / len(decisions) if decisions else nan,
            "ativacoes_por_minuto": dd / minutes if minutes else nan}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group in sorted({r["group"] for r in records}):
        rows = [r for r in records if r["group"] == group]
        block: dict[str, Any] = {"n": len(rows)}
        for label in LABELS:
            sel = [r for r in rows if r["label"] == label]
            if not sel:
                continue
            block[label] = {"n": len(sel), "ativacao": float(np.mean([r["activated"] for r in sel])),
                            "ativacao_politica_baixa_ndd": float(np.mean([r["activated_low_policy"] for r in sel])),
                            "sem_fala_detectada": float(np.mean([r["utterances"] == 0 for r in sel])),
                            "falas_por_item": float(np.mean([r["utterances"] for r in sel]))}
            if label == FREE_SPEECH:
                block[label].update(_per_utterance(sel))
        latencies = [x for r in rows for x in r["latencies_s"]]
        if latencies:
            block["latencia_s"] = {"p50": float(np.percentile(latencies, 50)),
                                   "p95": float(np.percentile(latencies, 95)), "n": len(latencies)}
        out[group] = block
    return out
