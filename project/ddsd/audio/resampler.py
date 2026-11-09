"""Reamostragem em fluxo, com estado entre blocos (soxr).

Reamostrar cada bloco isoladamente (`scipy.signal.resample_poly`) cria
descontinuidades nas bordas. Com `soxr.ResampleStream`, o resultado de blocos
de qualquer tamanho é igual ao da reamostragem do sinal inteiro, a menos do
atraso do filtro, que é entregue no `flush`.
"""

from __future__ import annotations

import numpy as np
import soxr


class StreamResampler:
    """Reamostrador mono `float32` de `in_rate` para `out_rate` (identidade se forem iguais)."""

    def __init__(self, in_rate: int, out_rate: int, quality: str = "HQ") -> None:
        self.in_rate = in_rate
        self.out_rate = out_rate
        self._stream = None if in_rate == out_rate else soxr.ResampleStream(
            in_rate, out_rate, 1, dtype="float32", quality=quality)

    @property
    def is_identity(self) -> bool:
        return self._stream is None

    def process(self, samples: np.ndarray, last: bool = False) -> np.ndarray:
        """Reamostra o próximo trecho do fluxo; `last=True` esvazia o atraso interno."""
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        if self._stream is None:
            return samples
        return self._stream.resample_chunk(samples, last=last)

    def reset(self) -> None:
        if self._stream is not None:
            self._stream = soxr.ResampleStream(self.in_rate, self.out_rate, 1, dtype="float32", quality="HQ")
