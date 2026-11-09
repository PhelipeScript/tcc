"""Semeadura de todos os geradores aleatórios (Python, NumPy, PyTorch CPU/CUDA/MPS)."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """Semeia `random`, `numpy`, PyTorch e `PYTHONHASHSEED`.

    O MPS não garante determinismo bit a bit (limitação documentada). A semente
    fixa a inicialização da cabeça, a ordem dos lotes e o dropout, mas somas em
    GPU podem variar na última casa.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
