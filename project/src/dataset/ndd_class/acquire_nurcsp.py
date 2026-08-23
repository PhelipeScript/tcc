"""
Aquisição da classe NDD a partir do corpus NURC-SP.

Fonte:
  nilc-nlp/CORAA-NURC-SP-Audio-Corpus — Hugging Face Hub.

Descrição:
  Este script obtém os metadados textuais necessários para a construção
  da classe NDD a partir do corpus NURC-SP (sotaque paulistano).

  São utilizados exclusivamente os CSVs de metadados da configuração
  'filtered', que já exclui segmentos com transcrição vazia. Os arquivos
  de áudio (.tar.gz) não são baixados, pois este projeto é puramente textual.

Licença:
  A licença não está explicitamente declarada no repositório no 
  Hugging Face Hub, porém o projeto TaRSila/C4AI, responsável pela
  disponibilização das versões do CORAA, identifica o CORAA NURC-SP sob
  a licença CC BY-NC-ND 4.0.

Uso:
  uv run python src/data/acquire_nurcsp.py --target 15000

Saída:
  Dataset textual contendo exemplos selecionados para a classe NDD.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_PENDING, Sample, normalize_text, write_jsonl

REPO_ID = "nilc-nlp/CORAA-NURC-SP-Audio-Corpus"
LICENSE = "CC BY-NC-ND 4.0 (HF Hub: not explicitly declared)"
SPLITS = ["train", "dev", "test"]
MIN_CHARS = 8

OUTPUT_PATH = DATA_PENDING / "ndd_nurcsp.jsonl"

def load_split(split: str) -> pd.DataFrame:
  """
    Carrega os metadados do split especificado do corpus NURC-SP. 
  """
  filename = f"filtered/audios_{split}_metadata.csv"
  path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=filename)
  return pd.read_csv(path, low_memory=False)


def clean(df: pd.DataFrame) -> pd.DataFrame:
  """
    Limpa os metadados do corpus NURC-SP, removendo exemplos inválidos e normalizando o texto.
  """
  df = df.copy()
  df = df.dropna(subset=["text"])
  df["text"] = df["text"].astype(str).map(normalize_text)
  df = df[df["text"].str.len() >= MIN_CHARS]
  return df


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", type=int, default=15_000, help="Volume alvo de amostras para a classe NDD.")
  parser.add_argument("--seed", type=int, default=42, help="Semente para o gerador de números aleatórios.")
  args = parser.parse_args()

  frames = []
  for split in SPLITS: 
    print(f"\033[33m[nurcsp] baixando metadados do split '{split}'...\033[0m")
    df = load_split(split)
    df = clean(df)
    df["hf_split"] = split
    frames.append(df)
    print(f"\033[34m[nurcsp]   {len(df)} linhas válidas após limpeza\033[0m")

  full = pd.concat(frames, ignore_index=True)
  print(f"\033[34m[nurcsp] total combinado: {len(full)} linhas\033[0m")
  print(f"\033[34m[nurcsp] distribuição por gênero de fala ('speech_genre'):\033[0m")
  print(full["speech_genre"].value_counts(dropna=False))

  if len(full) > args.target:
    frac = args.target / len(full)
    full = full.groupby("speech_genre", group_keys=False, dropna=False).apply(
      lambda g: g.sample(frac=frac, random_state=args.seed)
    )
    print(f"\033[33m[nurcsp] amostrado para {len(full)} linhas (estratificado por gênero de fala)\033[0m")

  samples = [
    Sample(
      text=row["text"],
      label="NDD",
      source="nurcsp",
      original_id=str(row.get("audio_name")),
      license=LICENSE,
      meta={
        "speech_genre": row.get("speech_genre"),
        "speech_style": row.get("speech_style"),
        "quality": row.get("quality"),
        "sex": row.get("sex"),
        "hf_split": row.get("hf_split"),
      },
    )
    for _, row in full.iterrows()
  ]

  n = write_jsonl(samples, OUTPUT_PATH)
  print(f"\033[32m[nurcsp] {n} amostras NDD escritas em {OUTPUT_PATH}\033[0m")


if __name__ == "__main__":
  main()
