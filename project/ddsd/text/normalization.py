"""Normalização textual única do projeto.

É a mesma função em três lugares: na preparação do corpus, no treino dos
classificadores e no pipeline em tempo real (cópia normalizada da transcrição
RAW do Whisper). Assim, a paridade entre treino e inferência vale por
construção, e não por convenção.

Ordem das operações (a ordem importa):
    1. decomposição Unicode NFKD e descarte das marcas de acento (categoria `Mn`);
    2. caixa baixa;
    3. todo caractere de pontuação (P*) ou símbolo (S*) vira espaço, para não
       colar palavras separadas por hífen ou barra;
    4. colapso de espaços e remoção das bordas.

NFKD, e não NFD, porque também dobra a tipografia de compatibilidade que NFD
deixaria intacta: `º` -> `o`, `³` -> `3`, ligaduras como `ﬁ` -> `fi`.

Exemplos:
    "Fecha essa aba, por favor!"  -> "fecha essa aba por favor"
    "não sei que horas são"       -> "nao sei que horas sao"
    "abre o pré-visualizador"     -> "abre o pre visualizador"
    "planilha do 3º trimestre"    -> "planilha do 3o trimestre"
"""

from __future__ import annotations

import unicodedata

#: Prefixos das categorias Unicode trocadas por espaço: pontuação (P*) e símbolo (S*).
_PUNCT_CATEGORY_PREFIXES: tuple[str, ...] = ("P", "S")


def strip_accents(text: str) -> str:
    """Remove diacríticos e dobra a tipografia de compatibilidade (NFKD sem `Mn`)."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def remove_punctuation(text: str) -> str:
    """Troca por espaço todo caractere de pontuação ou símbolo.

    A troca é por espaço, e não por string vazia, para que "pré-visualizador"
    vire duas palavras em vez de uma palavra inexistente.
    """
    return "".join(
        " " if unicodedata.category(ch).startswith(_PUNCT_CATEGORY_PREFIXES) else ch
        for ch in text
    )


def collapse_whitespace(text: str) -> str:
    """Colapsa qualquer sequência de espaços em branco e remove as bordas."""
    return " ".join(text.split())


def normalize_text(text: str) -> str:
    """Normalização completa: sem acento, minúscula, sem pontuação, espaços colapsados.

    Pode devolver string vazia (entrada só com pontuação ou espaços).
    """
    text = strip_accents(text)
    text = text.lower()
    text = remove_punctuation(text)
    return collapse_whitespace(text)


def word_count(text: str) -> int:
    """Número de palavras separadas por espaço (use sobre o texto já normalizado)."""
    return len(text.split())
