"""
Descrição:
    Amostragem estratificada do conjunto de teste para o experimento de ASR
    simulado — sintetizar e transcrever os 23.747 exemplos do teste seria
    caro (rede para o TTS, tempo de Whisper); uma amostra estratificada por
    `label + source` preserva a composição do teste original em escala menor.

Uso:
    from src.asr_eval.sampling import sample_test_set
    sample = sample_test_set(config)
"""

from __future__ import annotations

import numpy as np

from src.asr_eval.config import AsrEvalConfig
from src.data_prep import console
from src.data_prep.jsonl_io import Record, read_jsonl


def sample_test_set(config: AsrEvalConfig) -> list[Record]:
    """
    Descrição:
        Lê `test.jsonl` e sorteia uma amostra estratificada por
        `label + source`, do mesmo tamanho relativo entre estratos que o
        teste completo.

        Se `sample_size` for menor que o número de estratos (18, em
        `label x source`), a estratificação é impossível — cai para
        amostragem simples sem reposição. Só acontece em testes de fumaça
        com amostra minúscula; a execução real (`sample_size` na casa das
        centenas) sempre estratifica.

    Args:
        config: Configuração do experimento.

    Retorna:
        Lista de `Record` amostrados (ordem embaralhada, não a ordem do
        arquivo).
    """
    records = list(read_jsonl(config.data_dir / "test.jsonl"))
    if config.sample_size >= len(records):
        return records

    strata = [f"{record.label}|{record.source}" for record in records]
    n_strata = len(set(strata))
    if config.sample_size < n_strata:
        console.warn(
            f"sample_size={config.sample_size} menor que o número de estratos "
            f"({n_strata}); amostragem simples, sem estratificação"
        )
        rng = np.random.default_rng(config.seed)
        indices = rng.choice(len(records), size=config.sample_size, replace=False)
        return [records[i] for i in indices]

    from sklearn.model_selection import train_test_split

    indices = list(range(len(records)))
    sampled_indices, _ = train_test_split(
        indices,
        train_size=config.sample_size,
        stratify=strata,
        random_state=config.seed,
    )
    return [records[i] for i in sampled_indices]
