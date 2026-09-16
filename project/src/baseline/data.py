"""
Descrição:
    Carregamento leve dos splits do dataset para os baselines.

    Usa `src.data_prep.jsonl_io.read_jsonl` diretamente em vez da biblioteca
    `datasets` (usada pelo `src.training`): os baselines não precisam de
    tokenização em Arrow nem de `DataLoader`, e evitar essa dependência mantém
    o carregamento de ~190 mil linhas em segundos.

Uso:
    from src.baseline.data import load_split
    train = load_split(config.data_dir / "train.jsonl")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data_prep.jsonl_io import read_jsonl
from src.training.config import LABEL2ID


@dataclass(slots=True)
class SplitData:
    """
    Descrição:
        Uma partição do dataset já convertida para os tipos que os baselines
        consomem diretamente.

    Atributos:
        texts: Texto normalizado de cada exemplo, na ordem do arquivo.
        labels: Rótulos como inteiros (0 = NDD, 1 = DD), mesma codificação de
            `src.training.config.LABEL2ID`.
        sources: Provedor gerador de cada exemplo, na mesma ordem.
    """

    texts: list[str]
    labels: np.ndarray
    sources: list[str]

    def __len__(self) -> int:
        return len(self.texts)


def load_split(
    path: Path, *, max_samples: int | None = None, seed: int = 42
) -> SplitData:
    """
    Descrição:
        Lê um arquivo JSONL de split e monta um `SplitData`.

    Args:
        path: Caminho do `train.jsonl`, `val.jsonl` ou `test.jsonl`.
        max_samples: Se informado, sorteia N exemplos (sem reposição) em vez
            de mantê-los todos — usado só para teste de fumaça, nunca para o
            treino/avaliação real. Os arquivos vêm agrupados por classe e
            fonte (Stage 7), então pegar os primeiros N sem sortear
            frequentemente cai em uma única classe.
        seed: Semente do sorteio, quando `max_samples` é usado.

    Retorna:
        `SplitData` com textos, rótulos e fontes. Na ordem do arquivo quando
        `max_samples` é `None`; embaralhado quando um subconjunto é sorteado.
    """
    texts: list[str] = []
    labels: list[int] = []
    sources: list[str] = []

    for record in read_jsonl(path):
        texts.append(record.text)
        labels.append(LABEL2ID[record.label])
        sources.append(record.source)

    if max_samples is not None and max_samples < len(texts):
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(texts), size=max_samples, replace=False)
        texts = [texts[i] for i in indices]
        labels = [labels[i] for i in indices]
        sources = [sources[i] for i in indices]

    return SplitData(texts=texts, labels=np.asarray(labels, dtype=int), sources=sources)
