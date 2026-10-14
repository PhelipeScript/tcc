"""
Descrição:
    Transcrição com n-best real via beam search, usando o `ctranslate2` por
    trás do `faster-whisper`.

    A API de alto nível `WhisperModel.transcribe()` só devolve a melhor
    hipótese por segmento — não expõe a lista de hipóteses do beam search.
    O `ctranslate2.models.Whisper.generate()` interno, porém, aceita
    `num_hypotheses`, que devolve as N melhores sequências do mesmo beam
    search (confirmado por inspeção: `help(ctranslate2.models.Whisper.generate)`
    lista `num_hypotheses: int = 1` como parâmetro nativo). Este módulo refaz,
    manualmente, os passos que `transcribe()` faz internamente antes de
    chamar `generate()` — decodificar áudio, extrair features, montar o
    prompt — para então pedir N hipóteses em vez de 1.

    Depende de atributos e funções internas do `faster_whisper`
    (`WhisperModel.encode`, `WhisperModel.get_prompt`, `Tokenizer`,
    `get_suppressed_tokens`, `pad_or_trim`), que não fazem parte da API
    pública documentada e podem mudar em versões futuras do pacote. A versão
    fixada no `pyproject.toml` é a que foi validada.

    Usado apenas em clipes curtos e sintéticos (segundos, não minutos): não
    há VAD nem chunking de janelas de 30s como em `podcast_eval.py` — cada
    clipe é uma única janela.

Uso:
    from src.asr_eval.transcribe_nbest import transcribe_nbest
    hypotheses = transcribe_nbest(model, Path("clipe.wav"), n=3)
"""

from __future__ import annotations

from pathlib import Path

from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio, pad_or_trim
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import get_suppressed_tokens


def transcribe_nbest(
    model: WhisperModel,
    audio_path: Path,
    n: int,
    *,
    language: str = "pt",
    beam_size: int = 5,
) -> list[str]:
    """
    Descrição:
        Transcreve um clipe curto e devolve as N melhores hipóteses do beam
        search, na ordem de score (a primeira é a hipótese 1-best).

    Args:
        model: Instância de `WhisperModel` já carregada.
        audio_path: Caminho do áudio (qualquer formato que o `ffmpeg`, usado
            por `decode_audio`, decodifique).
        n: Número de hipóteses a devolver.
        language: Código de idioma forçado (evita a detecção automática, que
            é instável em clipes de poucas palavras).
        beam_size: Tamanho do beam search. Deve ser >= `n`, já que
            `num_hypotheses` não pode exceder o número de sequências do beam.

    Retorna:
        Lista com `n` transcrições (ou menos, se o beam produzir hipóteses
        duplicadas que o `ctranslate2` colapsa).

    Levanta:
        ValueError: Se `n` for maior que `beam_size`.
    """
    if n > beam_size:
        raise ValueError(f"n ({n}) não pode exceder beam_size ({beam_size})")

    waveform = decode_audio(str(audio_path), sampling_rate=model.feature_extractor.sampling_rate)
    features = model.feature_extractor(waveform)
    features = pad_or_trim(features, model.feature_extractor.nb_max_frames)
    encoder_output = model.encode(features)

    tokenizer = Tokenizer(
        model.hf_tokenizer, model.model.is_multilingual, task="transcribe", language=language
    )
    prompt = model.get_prompt(tokenizer, previous_tokens=[], without_timestamps=True)
    suppressed_tokens = get_suppressed_tokens(tokenizer, (-1,))

    results = model.model.generate(
        encoder_output,
        [prompt],
        beam_size=beam_size,
        num_hypotheses=n,
        length_penalty=1.0,
        suppress_blank=True,
        suppress_tokens=suppressed_tokens,
        return_scores=True,
    )
    hypotheses = results[0].sequences_ids[:n]
    return [tokenizer.decode(tokens).strip() for tokens in hypotheses]
