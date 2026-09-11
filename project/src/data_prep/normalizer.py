"""
Descrição:
    Normalização textual do dataset DDSD.

    A normalização aplicada é agressiva e intencional: o sistema final recebe a
    saída de um ASR, que não produz pontuação nem caixa confiável. Normalizar
    treino, validação e teste da mesma forma mantém a coerência e aproxima o
    texto da entrada real.

    Ordem das operações (importa):
        1. decomposição Unicode NFKD e descarte das marcas de acento
        2. caixa baixa
        3. remoção de toda pontuação e símbolo, substituídos por espaço para não 
            colar palavras separadas por hífen ou barra
        4. colapso de espaços múltiplos e remoção de espaços nas bordas

    Usa NFKD (decomposição de *compatibilidade*), e não NFD, de propósito. NFD
    resolve apenas acentos canônicos; NFKD também dobra a tipografia
    equivalente que aparece em texto real e que NFD deixaria intacta:
    `º` (U+00BA, categoria `Lo`) -> `o`, `³` (U+00B3, categoria `No`) -> `3`,
    ligaduras como `ﬁ` -> `fi`. Sem isso, `"3º trimestre"` sobreviveria com um
    caractere não-ASCII em um corpus supostamente sem acentuação.

    Exemplo:
        "Fecha essa aba, por favor!"  -> "fecha essa aba por favor"
        "não sei que horas são"       -> "nao sei que horas sao"
        "abre o pré-visualizador"     -> "abre o pre visualizador"
        "planilha do 3º trimestre"    -> "planilha do 3o trimestre"

Uso:
    from src.data_prep.normalizer import normalize_text, is_valid_length
"""

from __future__ import annotations

import unicodedata

from src.data_prep.config import MIN_CHARS, MIN_WORDS

#: Categorias Unicode removidas por `remove_punctuation`: toda pontuação (P*) e
#: todo símbolo (S*). Cobre `. , ; : ! ? " ' ( ) - _ … – — ’ “ ” « » / \ + = * # @`
#: e emojis, sem depender de uma lista fixa de caracteres ASCII.
_PUNCT_CATEGORY_PREFIXES: tuple[str, ...] = ("P", "S")


def strip_accents(text: str) -> str:
    """
    Descrição:
        Remove os diacríticos do texto e dobra a tipografia de compatibilidade,
        preservando as letras-base.

        Decompõe em NFKD (separando a letra da marca de acento e resolvendo
        equivalências de compatibilidade) e descarta os caracteres da categoria
        Unicode `Mn` (mark, nonspacing).

    Args:
        text: Texto de entrada.

    Retorna:
        Texto sem acentos. Exemplos: `"ação"` -> `"acao"`, `"você"` -> `"voce"`,
        `"3º"` -> `"3o"`, `"m³"` -> `"m3"`.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def remove_punctuation(text: str) -> str:
    """
    Descrição:
        Substitui por espaço todo caractere de pontuação ou símbolo.

        A substituição é por espaço — e não por string vazia — para que
        `"pré-visualizador"` se torne `"pré visualizador"` em vez de
        `"prévisualizador"`, preservando a contagem de palavras.

    Args:
        text: Texto de entrada.

    Retorna:
        Texto contendo apenas letras, dígitos e espaços (possivelmente múltiplos).
    """
    return "".join(
        " " if unicodedata.category(ch).startswith(_PUNCT_CATEGORY_PREFIXES) else ch
        for ch in text
    )


def collapse_whitespace(text: str) -> str:
    """
    Descrição:
        Colapsa qualquer sequência de espaços em branco em um único espaço e
        remove os espaços das bordas.

    Args:
        text: Texto de entrada.

    Retorna:
        Texto com espaçamento normalizado.
    """
    return " ".join(text.split())


def normalize_text(text: str) -> str:
    """
    Descrição:
        Aplica a normalização completa do dataset: sem acento, caixa baixa, sem
        pontuação, espaços colapsados.

        Esta é a função usada como chave de deduplicação em todos os estágios.

    Args:
        text: Texto original do provedor.

    Retorna:
        Texto normalizado. Pode ser string vazia se a entrada só continha
        pontuação ou espaços — nesse caso `is_valid_length` a rejeita.
    """
    text = strip_accents(text)
    text = text.lower()
    text = remove_punctuation(text)
    return collapse_whitespace(text)


def word_count(text: str) -> int:
    """
    Descrição:
        Conta as palavras do texto, separando por espaço em branco.

    Args:
        text: Texto, preferencialmente já normalizado.

    Retorna:
        Número de palavras.
    """
    return len(text.split())


def is_valid_length(text: str) -> bool:
    """
    Descrição:
        Verifica se o texto normalizado atende ao filtro de tamanho definido para
        o dataset: mais de `MIN_CHARS` caracteres **e** ao menos `MIN_WORDS`
        palavras.

    Args:
        text: Texto já normalizado.

    Retorna:
        `True` se o texto deve ser mantido no dataset.
    """
    return len(text) > MIN_CHARS and word_count(text) >= MIN_WORDS


def is_ascii(text: str) -> bool:
    """
    Descrição:
        Verifica se o texto contém apenas caracteres ASCII.

        Após a normalização, todo português brasileiro legítimo é ASCII: os
        acentos foram removidos e a tipografia de compatibilidade foi dobrada.
        O que sobra não-ASCII são scripts estrangeiros vazados pelos modelos
        geradores — chinês, cirílico, árabe, coreano — em frases que, no resto,
        estão em português.

    Args:
        text: Texto já normalizado.

    Retorna:
        `True` se o texto for inteiramente ASCII.
    """
    return text.isascii()


def rejection_reason(text: str, *, require_ascii: bool = True) -> str | None:
    """
    Descrição:
        Avalia se o texto normalizado deve ser descartado e, em caso positivo,
        informa o motivo. Concentra em um só lugar todos os critérios de
        exclusão do Stage 2, para que o relatório os contabilize separadamente.

    Args:
        text: Texto já normalizado.
        require_ascii: Se `True`, rejeita texto com script estrangeiro residual.

    Retorna:
        `None` se o texto deve ser mantido; caso contrário, o identificador do
        motivo: `"tamanho"` ou `"nao_ascii"`.
    """
    if not is_valid_length(text):
        return "tamanho"
    if require_ascii and not is_ascii(text):
        return "nao_ascii"
    return None


def describe_filter() -> str:
    """
    Descrição:
        Descreve o filtro de tamanho em texto legível, para uso nos relatórios.

    Retorna:
        Descrição do critério vigente, ex.: `"len > 5 e >= 2 palavras"`.
    """
    return f"len > {MIN_CHARS} e >= {MIN_WORDS} palavras"
