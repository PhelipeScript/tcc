"""
Descrição:
    Pacote de preparação do dataset DDSD (DD vs NDD) para o TCC2.

    Consolida os corpora sintéticos gerados por 9 provedores de LLM em um único
    dataset limpo, normalizado, deduplicado, balanceado e particionado em
    treino/validação/teste (80/10/10).

    O pipeline é dividido em 7 estágios independentes e reexecutáveis, cada um em
    seu próprio módulo `stageN_*.py`. Cada estágio lê o diretório do estágio
    anterior e escreve no seu, de modo que qualquer etapa pode ser reprocessada
    sem refazer as anteriores.

Uso:
    python -m src.data_prep.run_pipeline
"""

from __future__ import annotations

__all__ = [
    "config",
    "console",
    "jsonl_io",
    "normalizer",
    "report",
]
