"""
Descrição:
    Análise de podcast a partir de um vídeo do YouTube: baixa o áudio, transcreve
    com Whisper, quebra a fala em blocos de N segundos e roda cada bloco no
    modelo DD vs NDD escolhido — revelando em que trechos do episódio o
    assistente de voz tenderia a "acordar".

    A quebra considera apenas segundos de fala, não tempo de relógio: o VAD do
    Whisper descarta silêncio e música, então "quebrar a cada N segundos de fala"
    ignorar pausas é exatamente o que o bloco faz.

    Melhorias em relação ao pedido original:
        - limiar congelado do melhor checkpoint (mesma metodologia do pipeline);
        - blocos DD ranqueados e exibidos com marcação de tempo (MM:SS) para
          conferência auditiva;
        - transcrição e predições salvas em JSONL + resumo JSON para inspeção;
        - opção de forçar o idioma (português por padrão) porque os modelos
          foram treinados para o português brasileiro.

Uso:
    uv run python -m src.training.podcast_eval                        # interativo
    uv run python -m src.training.podcast_eval "URL" --model debertinha --secs 30
    uv run python -m src.training.podcast_eval "URL" --model bertimbau --secs 15 \
        --whisper-size medium --language auto
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import torch

from src.data_prep import console
from src.data_prep.normalizer import normalize_text
from src.training.config import DEFAULT_OUTPUT_DIR
from src.training.predict_interactive import MODEL_KEYS, ModelSession, ask_model

#: Diretório onde episódios analisados são salvos.
DEFAULT_ANALYSIS_DIR = DEFAULT_OUTPUT_DIR / "podcast_analysis"


def fmt_ts(seconds: float) -> str:
    """
    Formata segundos como `MM:SS` (ou `HH:MM:SS` para falas longas).

    Args:
        seconds: Duração em segundos (float).

    Retorna:
        Marcação de tempo navegável no player do YouTube.
    """
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def clip(text: str, limit: int = 64) -> str:
    """
    Trunca um trecho longo para a tabela de análise.

    Args:
        text: Texto a encurtar.
        limit: Comprimento máximo.

    Retorna:
        Texto truncado com reticências, se necessário.
    """
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def slugify(title: str, limit: int = 40) -> str:
    """
    Converte o título do vídeo em um slug seguro para nome de diretório.

    Args:
        title: Título do vídeo.
        limit: Tamanho máximo do slug.

    Retorna:
        Slug contendo apenas letras, dígitos e sublinhados.
    """
    plain = "".join(
        ch for ch in unicodedata.normalize("NFKD", title) if not unicodedata.category(ch).startswith("M")
    )
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", plain.lower()).strip("_")
    return slug[:limit].rstrip("_") or "episodio"


def download_audio(url: str, analysis_dir: Path) -> tuple[Path, dict]:
    """
    Baixa apenas o áudio do vídeo do YouTube.

    Args:
        url: URL do vídeo (ou ID).
        analysis_dir: Diretório onde o áudio será salvo.

    Retorna:
        Tupla `(caminho_do_audio, metadados_do_video)`.

    Levanta:
        RuntimeError: Se o yt-dlp não conseguir encontrar ou baixar o vídeo.
    """
    import yt_dlp

    analysis_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(analysis_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "noprogress": True,
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as exc:
        raise RuntimeError(f"não consegui baixar {url}: {exc}") from exc

    video_id = info.get("id")
    files = list(analysis_dir.glob(f"{video_id}.*"))
    if not files:
        raise RuntimeError(f"yt-dlp baixou, mas não achei o arquivo com id {video_id}")
    audio = max(files, key=lambda p: p.stat().st_size)
    return audio, info


def transcribe(audio_path: Path, size: str, language: str | None) -> tuple[list[dict], str, float | None]:
    """
    Transcreve o áudio com Whisper (via faster-whisper), com VAD ativado.

    Args:
        audio_path: Áudio baixado.
        size: Tamanho do modelo Whisper (`tiny` a `large-v3`).
        language: Idioma a forçar (`None` faz auto-detecção).

    Retorna:
        Tupla `(segmentos, idioma_detectado, duracao_do_audio_ou_None)`.

    Levanta:
        RuntimeError: Se o modelo Whisper não puder ser carregado.
    """
    from faster_whisper import WhisperModel

    use_cuda = torch.cuda.is_available()
    compute = "float16" if use_cuda else "int8"
    device = "cuda" if use_cuda else "cpu"
    console.info(f"carregando Whisper {size} ({device} {compute})…")
    try:
        whisper = WhisperModel(size, device=device, compute_type=compute)
    except Exception as exc:  # noqa: BLE001 - qualquer falha do ctranslate2/HF é terminal aqui
        raise RuntimeError(f"não consegui carregar o Whisper {size}: {exc}") from exc

    console.info(f"transcrevendo {audio_path.name} (VAD ligado)…")
    try:
        generator, info = whisper.transcribe(
            str(audio_path), language=language, beam_size=5, vad_filter=True
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"falha durante a transcrição: {exc}") from exc

    segments = [
        {"text": seg.text.strip(), "start": seg.start, "end": seg.end}
        for seg in generator
        if seg.text.strip()
    ]
    return segments, info.language, info.duration


def chunk_by_speech(segments: list[dict], seconds: int) -> list[dict]:
    """
    Agrupa os segmentos em blocos acumulando segundos de fala.

    Args:
        segments: Segmentos do Whisper, já com `text`, `start` e `end`.
        seconds: Alvo de segundos de fala por bloco.

    Retorna:
        Lista de blocos com `start`, `end`, `dur` e `texts`.
    """
    chunks: list[dict] = []
    current: dict | None = None
    for seg in segments:
        dur = seg["end"] - seg["start"]
        if current is None:
            current = {
                "start": seg["start"],
                "end": seg["end"],
                "dur": dur,
                "texts": [seg["text"]],
            }
        else:
            current["texts"].append(seg["text"])
            current["end"] = seg["end"]
            current["dur"] += dur
        if current["dur"] >= seconds:
            chunks.append(current)
            current = None
    if current:
        chunks.append(current)
    return chunks


def predict_chunks(chunks: list[dict], session: ModelSession) -> list[dict]:
    """
    Roda cada bloco no modelo escolhido, com a mesma normalização do pipeline.

    Args:
        chunks: Blocos produzidos por `chunk_by_speech`.
        session: Sessão do modelo carregada.

    Retorna:
        Blocos com `idx`, `text`, `norm`, `pred` e `prob` (sem blocos vazios).
    """
    records: list[dict] = []
    for idx, chunk in enumerate(chunks):
        text = " ".join(chunk["texts"])
        norm = normalize_text(text)
        if not norm:
            continue
        pred, prob = session.predict(norm)
        records.append({**chunk, "idx": len(records), "text": text, "norm": norm, "pred": pred, "prob": prob})
    return records


def save_artifacts(
    analysis_dir: Path,
    video: dict,
    segments: list[dict],
    records: list[dict],
    summary: dict,
) -> None:
    """
    Persiste a transcrição, as predições e o resumo do episódio.

    Args:
        analysis_dir: Diretório da análise.
        video: Metadados do vídeo (título, id, URL).
        segments: Segmentos do Whisper.
        records: Predições por bloco.
        summary: Resumo textual das métricas de acerto.
    """
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "transcript.jsonl").write_text(
        "\n".join(json.dumps(seg, ensure_ascii=False) for seg in segments) + "\n",
        encoding="utf-8",
    )
    (analysis_dir / "chunks.jsonl").write_text(
        "\n".join(json.dumps(rec, ensure_ascii=False) for rec in records) + "\n",
        encoding="utf-8",
    )
    (analysis_dir / "summary.json").write_text(
        json.dumps({**summary, "video": video}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def report(records: list[dict], session: ModelSession, speech_total: float, top: int = 5) -> dict:
    """
    Imprime a timeline de ativações e o resumo da análise.

    Args:
        records: Predições por bloco.
        session: Sessão do modelo (informativo do limiar).
        speech_total: Total de segundos de fala transcritos.
        top: Quantos blocos DD mais confiantes exibir.

    Retorna:
        Resumo com contagens para ser salvo em `summary.json`.
    """
    console.header(f"Timeline — {session.name}")
    console.info(f"limiar de decisão: {session.threshold:.3f}  blocos: {len(records)}")
    if not records:
        console.warn("nenhum bloco transcrito com texto válido")
        return {}

    console.table(
        ["bloco", "tempo", "fala", "p(DD)", "decisão", "trecho"],
        [
            [
                f"[{rec['idx']:02d}]",
                f"{fmt_ts(rec['start'])}–{fmt_ts(rec['end'])}",
                f"{rec['dur']:.0f}s",
                f"{rec['prob']:.4f}",
                rec["pred"],
                clip(rec["text"]),
            ]
            for rec in records
        ],
    )

    dd_records = [rec for rec in records if rec["pred"] == "DD"]
    dd_speech = sum(rec["dur"] for rec in dd_records)
    dd_records.sort(key=lambda rec: rec["prob"], reverse=True)

    console.header("Blocos em que o assistente acordaria (DD) — mais confiantes")
    console.info(f"{len(dd_records)} de {len(records)} blocos ({100 * len(dd_records) / len(records):.0f}%)")
    console.info(f"{dd_speech:.0f}s de {speech_total:.0f}s de fala ({100 * dd_speech / speech_total:.1f}%)")
    for rec in dd_records[:top]:
        console.info(
            f"  {fmt_ts(rec['start'])}  p(DD)={rec['prob']:.4f}  | {clip(rec['text'], 90)}"
        )
    if not dd_records:
        console.detail("(nenhum bloco despertou o assistente)")

    return {
        "model_name": session.key,
        "decision_threshold": session.threshold,
        "blocos": len(records),
        "blocos_dd": len(dd_records),
        "segundos_fala_total": round(speech_total, 1),
        "segundos_fala_dd": round(dd_speech, 1),
    }


def build_parser() -> argparse.ArgumentParser:
    """Monta o parser da linha de comando."""
    parser = argparse.ArgumentParser(
        prog="podcast_eval",
        description=(
            "Baixa um podcast do YouTube, transcreve e roda os blocos de fala "
            "no modelo DD vs NDD escolhido."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("url", nargs="?", default=None, help="URL do vídeo (ou ID)")
    parser.add_argument(
        "--model",
        choices=MODEL_KEYS,
        default=None,
        help="modelo DD vs NDD a usar (padrão: menu interativo)",
    )
    parser.add_argument(
        "--secs",
        type=int,
        default=30,
        help="segundos de fala por bloco",
    )
    parser.add_argument(
        "--whisper-size",
        choices=("tiny", "base", "small", "medium", "large-v3"),
        default="small",
        help="tamanho do modelo Whisper de transcrição",
    )
    parser.add_argument(
        "--language",
        default="pt",
        help="idioma forçado no Whisper (`auto` para detectar)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="quantos blocos DD mais confiantes exibir",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_ANALYSIS_DIR,
        help="diretório raiz das análises",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada: baixa, transcreve, quebra por fala e prediz."""
    args = build_parser().parse_args(argv)

    url = args.url
    if not url:
        url = input("URL do YouTube: ").strip()
        if not url:
            console.error("nenhuma URL informada")
            return 1

    model_key = args.model
    if model_key is None:
        while True:
            chosen = ask_model()
            if chosen is None:
                return 0
            model_key = chosen
            break

    console.header(f"Analisando podcast — modelo {model_key}")
    session = ModelSession.load(model_key)

    analysis_dir = args.out
    audio, video = download_audio(url, analysis_dir)
    console.ok(f"áudio salvo em {audio}")

    language = None if args.language == "auto" else args.language
    segments, detected, duration = transcribe(audio, args.whisper_size, language)
    console.ok(
        f"{len(segments)} segmentos, idioma {detected}",
    )
    speech_total = sum(seg["end"] - seg["start"] for seg in segments)
    console.info(f"fala detectada: {speech_total:.0f}s de {fmt_ts(duration or 0)} de áudio")

    chunks = chunk_by_speech(segments, args.secs)
    console.info(f"{len(chunks)} blocos de ~{args.secs}s de fala")
    records = predict_chunks(chunks, session)
    console.ok(f"{len(records)} blocos preditos")

    summary = report(records, session, speech_total, top=args.top)

    episode_dir = analysis_dir / f"{video.get('id')}_{slugify(video.get('title') or '')}"
    config_basedos = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    save_artifacts(
        episode_dir,
        {
            "title": video.get("title"),
            "id": video.get("id"),
            "url": url,
            "duration_s": video.get("duration"),
        },
        segments,
        records,
        {"config": config_basedos, **summary},
    )
    console.ok(f"artefatos salvos em {episode_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())