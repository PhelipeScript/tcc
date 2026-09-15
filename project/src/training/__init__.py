"""
Descrição:
    Pacote de fine-tuning dos encoders em português para a tarefa DDSD
    (device-directed vs non-device-directed).

    Treina três modelos sobre o mesmo dataset produzido por `src.data_prep`:

    - **BERTimbau Base** — `neuralmind/bert-base-portuguese-cased`
    - **Albertina PT-BR** — `PORTULAN/albertina-100m-portuguese-ptbr-encoder`
    - **DeBERTinha** — `sagui-nlp/debertinha-ptbr-xsmall`

    Toda a lógica de treino, avaliação e geração de artefatos vive em módulos
    compartilhados; cada modelo tem apenas um script fino que declara sua
    configuração. Trocar de modelo é editar um `ModelConfig`, nunca duplicar
    código de treino.

Uso:
    python -m src.training.train_bertimbau
    python -m src.training.train_albertina
    python -m src.training.train_debertinha
    python -m src.training.train_all
"""

from __future__ import annotations

__all__ = [
    "config",
    "data",
    "engine",
    "hardware",
    "metrics",
    "modeling",
    "reporting",
    "seeding",
]
