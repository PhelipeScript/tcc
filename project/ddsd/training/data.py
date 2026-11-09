"""Carregamento e tokenização das partições do corpus para o `Trainer`.

- Tokeniza sem padding: o `DataCollatorWithPadding` faz o padding por lote.
  Com enunciados curtos, isso reduz o custo por passo várias vezes sem mudar o
  resultado.
- `pad_to_multiple_of=8` alinha os lotes para os kernels de precisão mista.
- Colunas de texto (`source`, `bin`...) não sobrevivem à tokenização, porque o
  colator tentaria convertê-las em tensor. Os metadados de que os relatórios
  precisam são guardados antes, em `SplitMetadata`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, PreTrainedTokenizerBase

from ddsd.training.config import LABEL2ID, ModelConfig

SPLIT_FILES: dict[str, str] = {"train": "train.jsonl", "val": "val.jsonl", "test": "test.jsonl"}
_KEEP_COLUMNS = ("labels",)
_META_FIELDS = ("text", "source", "origin", "bin", "words")


@dataclass(frozen=True)
class SplitMetadata:
    """Campos por exemplo guardados antes da tokenização (na ordem do arquivo)."""

    text: list[str]
    source: list[str]
    origin: list[str]
    bin: list[str]
    words: list[int]


def load_splits(data_dir: Path, text_field: str = "text") -> DatasetDict:
    """Lê `train/val/test.jsonl` do corpus e cria a coluna numérica `labels`."""
    missing = [n for n, f in SPLIT_FILES.items() if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"partições ausentes em {data_dir}: {missing} "
                                "(rode `ddsd corpus build` e `ddsd corpus freeze`)")
    dataset = load_dataset("json", data_files={n: str(data_dir / f) for n, f in SPLIT_FILES.items()})
    if text_field not in dataset["train"].column_names:
        raise ValueError(f"campo {text_field!r} não existe; disponíveis: {dataset['train'].column_names}")
    dataset = dataset.map(lambda b: {"labels": [LABEL2ID[x] for x in b["label"]]}, batched=True,
                          desc="rótulos")
    if text_field != "text":
        dataset = dataset.remove_columns("text").rename_column(text_field, "text")
    return dataset


def subsample(dataset: Dataset, limit: int | None, seed: int) -> Dataset:
    """Até `limit` exemplos, de forma determinística (só para teste de fumaça)."""
    if limit is None or limit >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(limit))


def metadata(split: Dataset) -> SplitMetadata:
    """Metadados do split na ordem dos exemplos."""
    columns = split.column_names
    return SplitMetadata(**{f: list(split[f]) if f in columns else [""] * len(split) for f in _META_FIELDS})


def build_tokenizer(model: ModelConfig) -> PreTrainedTokenizerBase:
    """Tokenizador rápido do checkpoint (obrigatório para a DeBERTinha, que não tem `spm.model`)."""
    tokenizer = AutoTokenizer.from_pretrained(model.checkpoint, **model.tokenizer_kwargs)
    if not tokenizer.is_fast:
        raise RuntimeError(f"tokenizador de {model.checkpoint} não é a versão rápida")
    return tokenizer


def tokenize_splits(dataset: DatasetDict, tokenizer: PreTrainedTokenizerBase, max_length: int) -> DatasetDict:
    """Tokeniza (trunca em `max_length`, sem padding) e remove tudo que o modelo não consome."""
    tokenized = dataset.map(lambda b: tokenizer(b["text"], truncation=True, max_length=max_length),
                            batched=True, desc="tokenizando")
    drop = [c for c in tokenized["train"].column_names
            if c not in _KEEP_COLUMNS and not c.startswith(("input_ids", "attention_mask", "token_type_ids"))]
    return tokenized.remove_columns(drop)


def build_collator(tokenizer: PreTrainedTokenizerBase) -> DataCollatorWithPadding:
    return DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)


def describe_splits(dataset: DatasetDict) -> dict[str, Any]:
    """Composição de cada partição por classe e por fonte."""
    summary: dict[str, Any] = {}
    for name, split in dataset.items():
        labels = np.asarray(split["labels"])
        sources = list(split["source"])
        summary[name] = {
            "n": len(split),
            "por_classe": {lab: int((labels == idx).sum()) for lab, idx in LABEL2ID.items()},
            "por_fonte": {s: sources.count(s) for s in sorted(set(sources))},
        }
    return summary


def token_length_report(tokenized: DatasetDict, max_length: int, sample: int = 20_000) -> dict[str, float]:
    """Distribuição do comprimento em subtokens no treino (confirma que `max_length` sobra)."""
    split = tokenized["train"]
    limit = min(sample, len(split))
    lengths = np.array([len(ids) for ids in split.select(range(limit))["input_ids"]])
    return {
        "n_amostrado": int(limit), "media": float(lengths.mean()), "p50": float(np.percentile(lengths, 50)),
        "p95": float(np.percentile(lengths, 95)), "p99": float(np.percentile(lengths, 99)),
        "max": int(lengths.max()), "max_length_configurado": max_length,
        "fracao_truncada": float((lengths >= max_length).mean()),
    }
