"""
Descrição:
    Análise de overfitting de calibração e significância estatística entre os
    modelos já treinados — não retreina nada, só lê os artefatos que
    `src.training` e `src.baseline` já gravaram em `outputs/`.

    Cobre a decisão de "documentar e ilustrar, sem retreinar" o padrão de
    overfitting observado (eval_loss sobe após a época 2-3 enquanto as
    métricas discretas de validação continuam estáveis), e substitui as "5
    execuções com sementes diferentes" da metodologia original por bootstrap
    pareado sobre o conjunto de teste fixo.

Uso:
    python -m src.analysis.run_analysis
"""

from __future__ import annotations

__all__ = ["config", "loss_curves", "per_source_viz", "calibration", "bootstrap"]
