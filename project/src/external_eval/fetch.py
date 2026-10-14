"""
Descrição:
    Busca das transcrições textuais dos 3 corpora externos, sempre sem
    decodificar áudio — só as colunas de texto são lidas.

    CORAA e NURC-SP publicam a transcrição em CSVs de metadados (lidos
    diretamente por URL, sem passar pela biblioteca `datasets`); TAGARELA
    publica em Parquet com uma coluna `audio` (bytes) ao lado da coluna de
    texto — lida via `pyarrow` selecionando só `sentence`, o que evita
    carregar os bytes de áudio e a dependência de decodificação
    (`torchcodec`) que a biblioteca `datasets` exigiria.

    Schemas confirmados por inspeção manual em 16/09/2026 (podem mudar se os
    mantenedores dos repositórios alterarem a estrutura):
        - CORAA (`gabrielrstan/CORAA-v1.1`): CSV `metadata_test_final.csv`,
          coluna `text`, coluna `speech_style` (`"Spontaneous Speech"` é o
          subconjunto mais próximo de conversa espontânea real).
        - NURC-SP (`nilc-nlp/CORAA-NURC-SP-Audio-Corpus`): CSV
          `filtered/audios_test_metadata.csv`, coluna `text`; 100% do split
          de teste já é `speech_style == "spontaneous speech"`.
        - TAGARELA (`freds0/TAGARELA`): Parquet `data/test-00000-of-00001.parquet`,
          coluna `sentence` — parágrafos longos de fala transcrita (podcasts/
          entrevistas), por isso são quebrados em sentenças antes da amostragem,
          para ficarem na mesma escala das frases curtas do corpus de treino.

Uso:
    from src.external_eval.fetch import fetch_corpus
    texts = fetch_corpus("coraa", sample_size=1000, seed=42)
"""

from __future__ import annotations

import re

import numpy as np

#: URLs dos arquivos de metadados/dados usados por cada corpus.
CORAA_TEST_CSV = (
    "https://huggingface.co/datasets/gabrielrstan/CORAA-v1.1/"
    "resolve/main/metadata_test_final.csv"
)
NURC_SP_TEST_CSV = (
    "https://huggingface.co/datasets/nilc-nlp/CORAA-NURC-SP-Audio-Corpus/"
    "resolve/main/filtered/audios_test_metadata.csv"
)
TAGARELA_TEST_PARQUET = (
    "https://huggingface.co/datasets/freds0/TAGARELA/"
    "resolve/main/data/test-00000-of-00001.parquet"
)

#: Padrão de quebra de sentença: fim de frase seguido de espaço.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _sample(texts: list[str], sample_size: int, seed: int) -> list[str]:
    """
    Descrição:
        Sorteia até `sample_size` textos, sem reposição.

    Args:
        texts: Lista de textos candidatos.
        sample_size: Tamanho desejado da amostra.
        seed: Semente do sorteio.

    Retorna:
        Lista amostrada (a lista inteira, se `sample_size >= len(texts)`).
    """
    if sample_size >= len(texts):
        return texts
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(texts), size=sample_size, replace=False)
    return [texts[i] for i in indices]


def fetch_coraa(sample_size: int, seed: int) -> list[str]:
    """
    Descrição:
        Busca transcrições de fala espontânea do CORAA v1.1 (split de teste).

    Args:
        sample_size: Tamanho da amostra.
        seed: Semente do sorteio.

    Retorna:
        Lista de transcrições, sem normalizar.
    """
    import pandas as pd

    frame = pd.read_csv(CORAA_TEST_CSV)
    frame = frame[frame["speech_style"] == "Spontaneous Speech"]
    texts = frame["text"].dropna().astype(str).tolist()
    return _sample(texts, sample_size, seed)


def fetch_nurc_sp(sample_size: int, seed: int) -> list[str]:
    """
    Descrição:
        Busca transcrições de fala espontânea do corpus NURC-SP (split de
        teste; 100% já é `speech_style == "spontaneous speech"`).

    Args:
        sample_size: Tamanho da amostra.
        seed: Semente do sorteio.

    Retorna:
        Lista de transcrições, sem normalizar.
    """
    import pandas as pd

    frame = pd.read_csv(NURC_SP_TEST_CSV)
    texts = frame["text"].dropna().astype(str).tolist()
    return _sample(texts, sample_size, seed)


def fetch_tagarela(sample_size: int, seed: int) -> list[str]:
    """
    Descrição:
        Busca sentenças do TAGARELA (split de teste), quebrando os parágrafos
        longos de fala transcrita em sentenças individuais.

        Lê só a coluna `sentence` do Parquet via `pyarrow`, sem tocar a
        coluna `audio` — evita decodificar os bytes de áudio embutidos no
        arquivo (que exigiriam `torchcodec`, dependência que este projeto não
        precisa instalar só para descartar o áudio em seguida).

    Args:
        sample_size: Tamanho da amostra.
        seed: Semente do sorteio.

    Retorna:
        Lista de sentenças, sem normalizar.
    """
    import fsspec
    import pyarrow.parquet as pq

    with fsspec.open(TAGARELA_TEST_PARQUET) as handle:
        table = pq.ParquetFile(handle).read(columns=["sentence"])

    sentences: list[str] = []
    for paragraph in table.column("sentence").to_pylist():
        if not paragraph:
            continue
        for piece in _SENTENCE_SPLIT_RE.split(paragraph.strip()):
            piece = piece.strip()
            if piece:
                sentences.append(piece)

    return _sample(sentences, sample_size, seed)


#: Despacho por nome de corpus, usado por `fetch_corpus`.
FETCHERS = {
    "coraa": fetch_coraa,
    "nurc_sp": fetch_nurc_sp,
    "tagarela": fetch_tagarela,
}


def fetch_corpus(name: str, sample_size: int, seed: int) -> list[str]:
    """
    Descrição:
        Busca uma amostra de transcrições de um corpus pelo nome.

    Args:
        name: Nome do corpus (`"coraa"`, `"nurc_sp"` ou `"tagarela"`).
        sample_size: Tamanho da amostra.
        seed: Semente do sorteio.

    Retorna:
        Lista de transcrições, sem normalizar.

    Levanta:
        KeyError: Se `name` não estiver em `FETCHERS`.
    """
    return FETCHERS[name](sample_size, seed)
