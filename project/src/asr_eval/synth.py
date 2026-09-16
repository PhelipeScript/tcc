"""
Descrição:
    Síntese de voz (TTS) para simular áudio real a partir do texto sintético
    do corpus, permitindo testar o pipeline ASR → classificador sem precisar
    de gravações humanas (que não existem para as frases geradas por LLM).

    Usa `edge-tts` (cliente não oficial do serviço de TTS da Microsoft):
    leve, não usa GPU/CPU pesada, e tem vozes PT-BR de boa qualidade. É um
    serviço de terceiros, dependente de rede e sem SLA — por isso a interface
    fica isolada nesta única função, para trocar de motor (ex.: `gTTS` como
    fallback) sem tocar no resto do pipeline, e o áudio gerado é cacheado em
    disco (idempotente): uma amostra já sintetizada nunca é regerada.

Uso:
    from src.asr_eval.synth import synthesize
    path = synthesize("fecha essa aba por favor", Path("clipe_0001.mp3"), voice="pt-BR-AntonioNeural")
"""

from __future__ import annotations

import asyncio
from pathlib import Path

#: Vozes PT-BR usadas na síntese, para variar timbre/sotaque entre exemplos
#: (evita que o Whisper se acostume com uma única voz sintética).
VOICES: tuple[str, ...] = (
    "pt-BR-AntonioNeural",
    "pt-BR-FranciscaNeural",
)


async def _synthesize_async(text: str, out_path: Path, voice: str) -> None:
    """
    Descrição:
        Gera o áudio de forma assíncrona (API nativa do `edge-tts`).

    Args:
        text: Texto a sintetizar.
        out_path: Caminho do arquivo de áudio de saída (`.mp3`).
        voice: Nome da voz do `edge-tts`.
    """
    import edge_tts

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(out_path))


def synthesize(text: str, out_path: Path, voice: str = VOICES[0]) -> Path:
    """
    Descrição:
        Sintetiza `text` em áudio, gravando em `out_path`. Se o arquivo já
        existir, não sintetiza de novo (cache em disco).

    Args:
        text: Texto a sintetizar (idealmente `text_raw`, com pontuação —
            TTS sem pontuação soa artificial e pode mudar a prosódia).
        out_path: Caminho do `.mp3` de saída; o diretório pai é criado se
            necessário.
        voice: Nome da voz do `edge-tts` (ver `VOICES`).

    Retorna:
        `out_path`.

    Levanta:
        RuntimeError: Se a síntese falhar (rede indisponível, serviço fora
            do ar, texto vazio).
    """
    if out_path.exists():
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        asyncio.run(_synthesize_async(text, out_path, voice))
    except Exception as exc:  # noqa: BLE001 - qualquer falha do serviço é terminal aqui
        raise RuntimeError(f"falha ao sintetizar {text!r} com a voz {voice}: {exc}") from exc

    return out_path
