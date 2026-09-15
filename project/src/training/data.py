"""
Descrição:
    Carregamento e tokenização das partições do dataset.

    Lê os três JSONL produzidos pelo Stage 7 do pipeline de dados e os converte
    em um `DatasetDict` do `datasets`, tokenizado e pronto para o `Trainer`.

    Duas decisões de desempenho que valem explicação:

    - **Tokenização sem padding.** A tokenização apenas trunca; o padding é
      aplicado por lote pelo `DataCollatorWithPadding`. Como os enunciados são
      curtos (mediana em torno de 10 palavras) e `max_length` é 128, o padding
      fixo desperdiçaria a maior parte de cada lote. Com padding dinâmico o
      comprimento médio por lote cai para perto de 32, reduzindo o custo por
      passo em várias vezes, sem qualquer efeito no resultado.

    - **`pad_to_multiple_of=8`.** Alinha o comprimento do lote a múltiplos de 8,
      condição para os tensor cores da GPU serem usados em precisão mista.

Uso:
    from src.training.data import load_splits, build_tokenizer, tokenize_splits
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, PreTrainedTokenizerBase

from src.data_prep import console
from src.training.config import LABEL2ID, ModelConfig

#: Nome do arquivo de cada partição dentro do diretório de dados.
SPLIT_FILES: dict[str, str] = {
    "train": "train.jsonl",
    "val": "val.jsonl",
    "test": "test.jsonl",
}

#: Colunas preservadas após a tokenização, além dos tensores de entrada.
#:
#: Apenas `labels`. Colunas de texto como `source` NÃO podem sobreviver à
#: tokenização: o `DataCollatorWithPadding` tenta converter em tensor tudo que
#: recebe, e falha em strings. Os provedores e os textos de que os relatórios
#: precisam são capturados do dataset ainda não tokenizado, em `engine.run`.
KEEP_COLUMNS: tuple[str, ...] = ("labels",)


def load_splits(data_dir: Path, text_field: str = "text") -> DatasetDict:
    """
    Descrição:
        Carrega as três partições do dataset e prepara a coluna de rótulo.

        A coluna usada como entrada do modelo é renomeada para `text`
        internamente, de modo que o resto do pipeline não precise saber se a
        origem foi o texto normalizado ou o original.

    Args:
        data_dir: Diretório com `train.jsonl`, `val.jsonl` e `test.jsonl`.
        text_field: Campo do JSONL usado como entrada: `"text"` (normalizado) ou
            `"text_raw"` (original, com acento e pontuação).

    Retorna:
        `DatasetDict` com as chaves `train`, `val` e `test`, cada uma com as
        colunas `text`, `labels`, `source` e `label`.

    Levanta:
        FileNotFoundError: Se alguma partição estiver ausente.
        ValueError: Se `text_field` não existir nos dados.
    """
    missing = [name for name, filename in SPLIT_FILES.items() if not (data_dir / filename).exists()]
    if missing:
        raise FileNotFoundError(
            f"partições ausentes em {data_dir}: {missing}. "
            "Rode o pipeline de dados: python -m src.data_prep.run_pipeline"
        )

    dataset = load_dataset(
        "json",
        data_files={name: str(data_dir / filename) for name, filename in SPLIT_FILES.items()},
    )

    available = dataset["train"].column_names
    if text_field not in available:
        raise ValueError(f"campo {text_field!r} não existe nos dados; disponíveis: {available}")

    def add_label_id(batch: dict[str, list[Any]]) -> dict[str, list[int]]:
        """
        Descrição:
            Converte o rótulo textual em índice numérico.

        Args:
            batch: Lote de exemplos com a coluna `label`.

        Retorna:
            Dicionário com a coluna `labels` numérica.
        """
        return {"labels": [LABEL2ID[label] for label in batch["label"]]}

    dataset = dataset.map(add_label_id, batched=True, desc="rotulos -> indices")

    if text_field != "text":
        # Substitui a coluna de entrada mantendo o nome `text` internamente.
        dataset = dataset.remove_columns("text").rename_column(text_field, "text")

    return dataset


def subsample(dataset: Dataset, limit: int | None, seed: int) -> Dataset:
    """
    Descrição:
        Reduz uma partição a no máximo `limit` exemplos, de forma determinística.
        Usada apenas em testes de fumaça.

    Args:
        dataset: Partição a reduzir.
        limit: Número máximo de exemplos. `None` devolve a partição intacta.
        seed: Semente da embaralhada.

    Retorna:
        Partição possivelmente reduzida.
    """
    if limit is None or limit >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(limit))


def build_tokenizer(model: ModelConfig) -> PreTrainedTokenizerBase:
    """
    Descrição:
        Instancia o tokenizador do checkpoint.

        Usa sempre o caminho rápido (`AutoTokenizer` com `use_fast` no padrão).
        Isso é obrigatório para o DeBERTinha: o repositório declara
        `DebertaV2Tokenizer` (classe lenta, baseada em SentencePiece) mas não
        contém `spm.model`, apenas `tokenizer.json` — forçar o tokenizador lento
        falharia.

    Args:
        model: Configuração do modelo.

    Retorna:
        Tokenizador carregado.
    """
    return AutoTokenizer.from_pretrained(model.checkpoint, **model.tokenizer_kwargs)


def tokenize_splits(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> DatasetDict:
    """
    Descrição:
        Tokeniza todas as partições, truncando em `max_length` e **sem** aplicar
        padding (que fica a cargo do colator, por lote).

    Args:
        dataset: `DatasetDict` carregado por `load_splits`.
        tokenizer: Tokenizador do modelo.
        max_length: Comprimento máximo em subtokens.

    Retorna:
        `DatasetDict` tokenizado, contendo os tensores de entrada mais as
        colunas de `KEEP_COLUMNS`.
    """

    def encode(batch: dict[str, list[Any]]) -> dict[str, Any]:
        """
        Descrição:
            Tokeniza um lote de textos.

        Args:
            batch: Lote com a coluna `text`.

        Retorna:
            Saída do tokenizador para o lote.
        """
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    tokenized = dataset.map(encode, batched=True, desc="tokenizando")

    # Remove tudo que o modelo não consome. Deixar colunas de texto aqui faz o
    # colator falhar ao montar o lote ("Unable to create tensor").
    drop = [
        column
        for column in tokenized["train"].column_names
        if column not in KEEP_COLUMNS and not column.startswith(("input_ids", "attention_mask", "token_type_ids"))
    ]
    return tokenized.remove_columns(drop)


def build_collator(tokenizer: PreTrainedTokenizerBase) -> DataCollatorWithPadding:
    """
    Descrição:
        Constrói o colator de padding dinâmico, alinhado a múltiplos de 8.

    Args:
        tokenizer: Tokenizador do modelo.

    Retorna:
        Colator pronto para o `Trainer`.
    """
    return DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)


def describe_splits(dataset: DatasetDict) -> dict[str, Any]:
    """
    Descrição:
        Resume a composição de cada partição por classe e por provedor.

    Args:
        dataset: `DatasetDict` carregado (antes ou depois da tokenização).

    Retorna:
        Dicionário `partição -> {n, por_classe, por_provedor}`.
    """
    summary: dict[str, Any] = {}
    for name, split in dataset.items():
        labels = split["labels"]
        sources = split["source"]
        summary[name] = {
            "n": len(split),
            "por_classe": {
                label: int(np.sum(np.asarray(labels) == index))
                for label, index in LABEL2ID.items()
            },
            "por_provedor": {
                provider: int(sources.count(provider)) for provider in sorted(set(sources))
            },
        }
    return summary


def token_length_report(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    sample: int = 20_000,
) -> dict[str, float]:
    """
    Descrição:
        Mede a distribuição de comprimento em subtokens no conjunto de treino.

        Serve para confirmar empiricamente que `max_length` é generoso e para
        estimar o ganho do padding dinâmico. Se o percentil 99 estiver muito
        abaixo de `max_length`, o padding fixo estaria desperdiçando computação.

    Args:
        dataset: `DatasetDict` tokenizado.
        tokenizer: Tokenizador (usado apenas para o nome no relatório).
        max_length: Comprimento máximo configurado.
        sample: Número de exemplos amostrados para a medição.

    Retorna:
        Dicionário com média e percentis do comprimento, e a fração truncada.
    """
    split = dataset["train"]
    limit = min(sample, len(split))
    lengths = np.array([len(ids) for ids in split.select(range(limit))["input_ids"]])
    return {
        "n_amostrado": int(limit),
        "media": float(lengths.mean()),
        "p50": float(np.percentile(lengths, 50)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
        "max_length_configurado": max_length,
        "fracao_truncada": float((lengths >= max_length).mean()),
    }


def log_splits(summary: dict[str, Any]) -> None:
    """
    Descrição:
        Imprime a composição das partições no terminal.

    Args:
        summary: Resumo produzido por `describe_splits`.
    """
    console.header("Dataset")
    rows = [
        [name, info["n"], *[info["por_classe"][label] for label in LABEL2ID]]
        for name, info in summary.items()
    ]
    console.table(["partição", "n", *LABEL2ID], rows)
