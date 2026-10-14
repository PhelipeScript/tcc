"""
Descrição:
    Baselines clássicos (TF-IDF + SVM linear calibrado / Regressão Logística)
    para a tarefa DDSD, exigidos pela metodologia do TCC como referência de
    comparação para o BERTimbau (Experimento 1).

    Roda sobre os mesmos splits produzidos por `src.data_prep`
    (`data/07_splits/{train,val,test}.jsonl`) e grava artefatos no mesmo
    layout de `src.training` (`outputs/baseline_<modelo>/`), reaproveitando
    `src.training.metrics` e `src.training.reporting` sem alterá-los.

Uso:
    python -m src.baseline.train_baselines
"""

from __future__ import annotations

__all__ = ["config", "data", "vectorize", "models", "tuning", "engine"]
