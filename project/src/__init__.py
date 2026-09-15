"""
Descrição:
    Raiz do código-fonte do TCC2 "Detecção de Intenção Implícita para Ativação
    de Assistentes de Voz em Ambientes Controlados".

    Subpacotes:
        dataset   — legado: geração do corpus sintético com LLMs
        data_prep — preparação do dataset (normalização, dedup, balanceamento, split)
        training  — fine-tuning de BERTimbau, Albertina e DeBERTinha

Uso:
    python -m src.data_prep.run_pipeline
    python -m src.training.train_all
"""

from __future__ import annotations
