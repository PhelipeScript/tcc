"""Silero VAD em ONNX (CPU, 1 thread): probabilidade de fala por bloco de 512 amostras a 16 kHz."""

from __future__ import annotations

import numpy as np
import torch

from ddsd.realtime.config import MODEL_RATE, VAD_CHUNK


class SileroVad:
    """Modelo com estado: os blocos devem chegar em ordem, sem lacunas (`reset` entre fluxos)."""

    def __init__(self) -> None:
        from silero_vad import load_silero_vad

        # O wrapper ONNX do pacote fixa 1 thread intra/inter-op, então não disputa CPU com o resto.
        self._model = load_silero_vad(onnx=True)

    def probability(self, chunk: np.ndarray) -> float:
        """P(fala) do bloco (exatamente 512 amostras `float32` a 16 kHz)."""
        if chunk.shape != (VAD_CHUNK,):
            raise ValueError(f"o Silero exige blocos de {VAD_CHUNK} amostras; recebeu {chunk.shape}")
        with torch.inference_mode():
            return float(self._model(torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float32)),
                                     MODEL_RATE).item())

    def reset(self) -> None:
        self._model.reset_states()
