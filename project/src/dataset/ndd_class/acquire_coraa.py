"""
Aquisição da classe NDD a partir do corpus CORAA ASR.

Fonte:
  gabrielrstan/CORAA-v1.1 — Hugging Face Hub.

Descrição:
  Este script obtém os metadados textuais necessários para a construção
  da classe NDD a partir do corpus CORAA. Os áudios não são baixados,
  pois este projeto é puramente textual.

Licença:
  A versão do dataset disponibilizada no Hugging Face Hub está marcada
  com licença "unknown". O corpus CORAA original é disponibilizado sob
  a licença CC BY-NC-ND 4.0, conforme indicado pelo projeto CORAA e sua
  publicação original.

  A licença do material disponibilizado no Hugging Face deve ser
  confirmada individualmente antes de qualquer redistribuição do
  conteúdo ou de um dataset derivado.

Uso:
  uv run python src/data/acquire_coraa.py --target 25000

Saída:
  Dataset textual contendo exemplos selecionados para a classe NDD.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DATA_PENDING, Sample, normalize_text, write_jsonl

REPO_ID = "gabrielrstan/CORAA-v1.1"
LICENSE = "CC BY-NC-ND 4.0 (HF Hub: unknown)"
SPLITS = ["train", "dev", "test"]
MIN_CHARS = 8

OUTPUT_PATH = DATA_PENDING / "ndd_coraa.jsonl"


def load_split(split: str) -> pd.DataFrame:
  """
    Carrega os metadados do split especificado do corpus CORAA. 
  """
  filename = f"metadata_{split}_final.csv"
  path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=filename)
  return pd.read_csv(path, low_memory=False)


def clean(df: pd.DataFrame) -> pd.DataFrame:
  """
    Limpa os metadados do corpus CORAA, removendo exemplos inválidos e normalizando o texto.
  """
  df = df.copy()
  df = df.dropna(subset=["text"])
  df["text"] = df["text"].astype(str).map(normalize_text)
  df = df[df["text"].str.len() >= MIN_CHARS]
  # Remove segmentos invalidados por consenso dos anotadores (mais votos contra que a favor)
  if "up_votes" in df.columns and "down_votes" in df.columns:
    invalid = df["down_votes"].fillna(0) > df["up_votes"].fillna(0)
    df = df[~invalid]
  return df


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", type=int, default=25_000, help="Volume alvo de amostras para a classe NDD.")
  parser.add_argument("--seed", type=int, default=42, help="Semente para o gerador de números aleatórios.")
  args = parser.parse_args()

  frames = []
  for split in SPLITS:
    print(f"\033[33m[coraa] baixando metadados do split '{split}'...\033[0m")
    df = load_split(split)
    df = clean(df)
    df["hf_split"] = split
    frames.append(df)
    print(f"\033[34m[coraa]   {len(df)} linhas válidas após limpeza\033[0m")

  full = pd.concat(frames, ignore_index=True)
  print(f"\033[34m[coraa] total combinado: {len(full)} linhas\033[0m")
  print(f"\033[34m[coraa] distribuição por sub-corpus ('dataset'):\033[0m")
  print(full["dataset"].value_counts(dropna=False))

  # amostragem estratificada por sub-corpus de origem, até o volume alvo
  if len(full) > args.target:
    frac = args.target / len(full)
    full = full.groupby("dataset", group_keys=False, dropna=False).apply(
      lambda g: g.sample(frac=frac, random_state=args.seed)
    )
    print(f"\033[33m[coraa] amostrado para {len(full)} linhas (estratificado por sub-corpus)\033[0m")

  samples = [
    Sample(
      text=row["text"],
      label="NDD",
      source="coraa",
      original_id=str(row.get("file_path")),
      license=LICENSE,
      meta={
        "sub_corpus": row.get("dataset"),
        "speech_genre": row.get("speech_genre"),
        "speech_style": row.get("speech_style"),
        "accent": row.get("accent"),
        "hf_split": row.get("hf_split"),
      },
    )
    for _, row in full.iterrows()
  ]

  n = write_jsonl(samples, OUTPUT_PATH)
  print(f"\033[32m[coraa] {n} amostras NDD escritas em {OUTPUT_PATH}\033[0m")

if __name__ == "__main__":
  main()
