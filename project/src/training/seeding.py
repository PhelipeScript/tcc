"""
Descrição:
    Semeadura de todos os geradores de números aleatórios envolvidos no treino.

    Sem isso, duas execuções do mesmo script com a mesma configuração produzem
    resultados diferentes: a inicialização do cabeçalho de classificação, a
    ordem dos lotes e o dropout todos consomem aleatoriedade. Para um TCC, o
    resultado precisa ser reproduzível.

Uso:
    from src.training.seeding import set_all_seeds
    set_all_seeds(42)
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

#: Comando que precisa ser exportado ANTES de importar o torch para que o modo
#: determinístico funcione com operações de cuBLAS.
DETERMINISM_ENV_HINT: str = "CUBLAS_WORKSPACE_CONFIG=:4096:8"


def set_all_seeds(seed: int) -> None:
    """
    Descrição:
        Semeia `random`, `numpy`, PyTorch (CPU e todas as GPUs) e a variável de
        ambiente `PYTHONHASHSEED`.

    Args:
        seed: Semente a aplicar.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_determinism(warn_only: bool = True) -> None:
    """
    Descrição:
        Ativa o modo determinístico do PyTorch, desligando os kernels
        não-determinísticos do cuDNN e a busca automática de algoritmos.

        Custa velocidade e não é necessário para reprodutibilidade prática — a
        semente já basta na maioria dos casos. Use apenas quando for preciso
        garantir resultado idêntico bit a bit.

    Args:
        warn_only: Se `True`, operações sem implementação determinística geram
            aviso em vez de exceção.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
