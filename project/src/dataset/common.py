"""Utilidades compartilhadas pelos scripts de construção do corpus DDSD"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Literal

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PENDING = REPO_ROOT / "data" / "pending"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"

Label = Literal["DD", "NDD", "AMB"]

@dataclass
class Sample:
  text: str
  label: Label
  source: str # coraa | nurcsp | tagarela | synthetic:<strategy>
  original_id: str | None = None
  license: str | None = None
  strategy: str | None = None # template_paraphrase | roleplay | dialogue (se sintético)
  model: str | None = None # backend LLM usado (se sintético)
  meta: dict = field(default_factory=dict)

  def to_json(self) -> str:
    return json.dumps(asdict(self), ensure_ascii=False, indent=2)

def write_jsonl(samples: Iterable[Sample], path: Path) -> int:
  """Escreve uma lista de amostras em um arquivo JSONL"""
  path.parent.mkdir(parents=True, exist_ok=True)
  n = 0
  with path.open("w", encoding="utf-8") as f:
    for s in samples:
      f.write(s.to_json() + "\n")
      n += 1
  return n

def append_jsonl(samples: Iterable[Sample], path: Path) -> int:
  """Adiciona uma lista de amostras em um arquivo JSONL"""
  path.parent.mkdir(parents=True, exist_ok=True)
  n = 0
  with path.open("a", encoding="utf-8") as f:
    for s in samples:
      f.write(s.to_json() + "\n")
      n += 1
  return n

def read_jsonl(path: Path) -> Iterator[Sample]:
  """Lê um arquivo JSONL e retorna um iterador de amostras"""
  if not path.exists():
    return
  with path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      d = json.loads(line)
      yield Sample(**d)

def normalize_text(text: str) -> str:
  """Normalização leve: colapsa espaços, remove espaços nas bordas."""
  return " ".join(text.split()).strip()
