"""
Aquisição da classe NDD a partir do corpus TAGARELA.

Fonte:
  freds0/TAGARELA — Hugging Face Hub.

Descrição:
  Este script obtém os dados textuais necessários para a construção da
  classe NDD a partir do corpus TAGARELA, composto por podcasts em
  português.

  O dataset completo possui aproximadamente 1,76 TB, incluindo áudio e
  texto, distribuídos em 1764 shards de treinamento. Como este projeto
  utiliza apenas informações textuais, são baixados somente os shards
  Parquet necessários. As colunas 'sentence' e 'path' são extraídas
  utilizando PyArrow, sem decodificação da coluna de áudio.

  Apesar disso, cada shard de treinamento contém os bytes de áudio
  incorporados no arquivo Parquet, resultando em aproximadamente 850 MB
  por shard. O número de shards deve ser ajustado de acordo com a
  disponibilidade de banda e armazenamento.

Licença:
  CC BY-NC-SA 4.0.

Uso:
  uv run python src/data/acquire_tagarela.py --target 15000 --num-train-shards 4

Saída:
  Dataset textual contendo exemplos selecionados para a classe NDD.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pandas as pd
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DATA_PENDING, Sample, normalize_text, write_jsonl

REPO_ID = "freds0/TAGARELA"
LICENSE = "CC BY-NC-SA 4.0"
MIN_CHARS = 8
TEST_SHARD = "data/test-00000-of-00001.parquet"
TRAIN_SHARD_TEMPLATE = "data/train-{i:05d}-of-01764.parquet"

OUTPUT_PATH = DATA_PENDING / "ndd_tagarela.jsonl"


def load_shard(filename: str):
  path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=filename)
  table = pq.ParquetFile(path).read(columns=["sentence", "path"])
  return table.to_pandas()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", type=int, default=15_000, help="Volume alvo de amostras para a classe NDD.")
  parser.add_argument(
    "--num-train-shards", 
    type=int, 
    default=4, 
    help="Quantos shards de treino baixar além do shard de teste (cada um ~850 MB)."
  )
  parser.add_argument("--seed", type=int, default=42, help="Semente para o gerador de números aleatórios.")
  args = parser.parse_args()

  all_rows = []

  print(f"\033[33m[tagarela] baixando shard de teste ({TEST_SHARD})...\033[0m")
  df_test = load_shard(TEST_SHARD)
  df_test["hf_split"] = "test"
  all_rows.append(df_test)
  print(f"\033[34m[tagarela]   {len(df_test)} linhas\033[0m")

  total = len(df_test)
  for i in range(args.num_train_shards):
    if total >= args.target:
      break
    shard_name = TRAIN_SHARD_TEMPLATE.format(i=i)
    print(f"\033[33m[tagarela] baixando shard de treino {i} ({shard_name})... (~850 MB)\033[0m")
    df_train = load_shard(shard_name)
    df_train["hf_split"] = "train"
    all_rows.append(df_train)
    total += len(df_train)
    print(f"\033[34m[tagarela]   {len(df_train)} linhas (acumulado: {total})\033[0m")

  full = pd.concat(all_rows, ignore_index=True)
  full = full.dropna(subset=["sentence"])
  full["text"] = full["sentence"].astype(str).map(normalize_text)
  full = full[full["text"].str.len() >= MIN_CHARS]
  print(f"\033[34m[tagarela] total válido após limpeza: {len(full)} linhas\033[0m")

  if len(full) > args.target:
    full = full.sample(n=args.target, random_state=args.seed)
    print(f"\033[33m[tagarela] amostrado para {len(full)} linhas\033[0m")

  samples = [
    Sample(
      text=row["text"],
      label="NDD",
      source="tagarela",
      original_id=str(row.get("path")),
      license=LICENSE,
      meta={
        "hf_split": row.get("hf_split"),
        "asr_generated_label": True,
      },
    )
    for _, row in full.iterrows()
  ]

  n = write_jsonl(samples, OUTPUT_PATH)
  print(f"\033[32m[tagarela] {n} amostras NDD escritas em {OUTPUT_PATH}\033[0m")

if __name__ == "__main__":
  main()
