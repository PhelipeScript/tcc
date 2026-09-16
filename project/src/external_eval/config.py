"""
Descrição:
    Configuração da validação externa.

Uso:
    from src.external_eval.config import ExternalEvalConfig
    config = ExternalEvalConfig()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.training.config import DEFAULT_OUTPUT_DIR

#: Nomes dos corpora suportados, na ordem de exibição.
CORPUS_NAMES: tuple[str, ...] = ("coraa", "nurc_sp", "tagarela")


@dataclass(frozen=True)
class ExternalEvalConfig:
    """
    Descrição:
        Configuração de uma execução da validação externa.

    Atributos:
        output_dir: Raiz de saída da validação externa.
        corpora: Corpora a avaliar (subconjunto de `CORPUS_NAMES`).
        sample_size: Exemplos sorteados por corpus, após a busca (antes do
            filtro de tamanho — a amostra final pode ficar um pouco menor).
        model_keys: Modelos de classificação avaliados.
        seed: Semente da amostragem.
        top_false_positives: Quantos falsos positivos mais confiantes salvar
            por corpus/modelo, para análise qualitativa.
    """

    output_dir: Path = DEFAULT_OUTPUT_DIR / "external_eval"
    corpora: tuple[str, ...] = CORPUS_NAMES
    sample_size: int = 1000
    model_keys: tuple[str, ...] = ("bertimbau", "albertina", "debertinha")
    seed: int = 42
    top_false_positives: int = 30
