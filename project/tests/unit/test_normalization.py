"""Normalização: exemplos documentados."""

from __future__ import annotations

import pytest

from ddsd.text.normalization import normalize_text, word_count


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Fecha essa aba, por favor!", "fecha essa aba por favor"),
        ("não sei que horas são", "nao sei que horas sao"),
        ("abre o pré-visualizador", "abre o pre visualizador"),
        ("planilha do 3º trimestre", "planilha do 3o trimestre"),
        ("  Ação   CORRETA\t\n", "acao correta"),
        ("m³ e ﬁm", "m3 e fim"),
        ("...!?", ""),
        ("e-mail/whatsapp", "e mail whatsapp"),
        ("emoji 😀 no meio", "emoji no meio"),
    ],
)
def test_documented_examples(raw: str, expected: str) -> None:
    assert normalize_text(raw) == expected


def test_idempotent() -> None:
    sample = "Você pode abrir o Chrome - e a aba do e-mail? Obrigado!!"
    once = normalize_text(sample)
    assert normalize_text(once) == once


def test_word_count() -> None:
    assert word_count("abre o navegador") == 3
    assert word_count("") == 0
