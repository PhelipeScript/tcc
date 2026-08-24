"""
Geração sintética das classes DD/AMB por meio de modelos de linguagem (LLM).

Descrição:
  Este script gera exemplos sintéticos para as classes DD (Directives /
  comandos diretos) e AMB (ambígua), no domínio de interação com
  computadores pessoais.

  São implementadas três estratégias de geração:

    1. template:
      Geração de paráfrases a partir de templates organizados por
      categoria de comando.

    2. roleplay:
      Geração contextual baseada em persona, contexto de utilização
      do computador e tom da interação.

    3. dialogue:
      Geração de diálogos sintéticos completos com turnos rotulados
      como DD, NDD ou AMB. Os turnos classificados como DD ou AMB
      são incorporados ao corpus, enquanto os turnos NDD são
      utilizados apenas para fornecer coerência contextual aos
      diálogos, uma vez que essa classe é obtida a partir de corpora
      públicos.

Backends:
  O script possui uma interface LLMBackend que permite utilizar diferentes
  modelos de linguagem. Atualmente são suportados os backends Gemini,
  Mistral e Gemma4 (execução local).

  Novos backends podem ser adicionados mediante a implementação da
  interface LLMBackend.

Retomada:
  A execução é retomável. Cada chamada de geração recebe um identificador
  determinístico calculado a partir da estratégia, dos parâmetros e do
  índice do lote.

  Os identificadores processados são registrados em:
      data/pending/synthetic_checkpoint.json

  Esse mecanismo evita chamadas duplicadas durante reexecuções e reduz
  o consumo desnecessário das APIs, especialmente em razão dos limites
  de requisições (rate limits) dos backends gratuitos.

Licença:
  Os exemplos gerados por este script são dados sintéticos produzidos por
  modelos de linguagem e não são provenientes diretamente dos corpora
  públicos utilizados para a classe NDD.

Uso:
  Configure a chave da API correspondente ao backend escolhido:

    export GEMINI_API_KEY=...
    # ou
    export MISTRAL_API_KEY=...
    # ou, para o backend local Gemma4 (via Ollama), inicie o servidor:
    export OLLAMA_HOST=...       # opcional, padrão http://localhost:11434
    export GEMMA_MODEL=...       # opcional, padrão gemma4:e4b
    ollama serve

  Execução em modo piloto:

    uv run python src/data/generate_synthetic.py \
      --backend gemma4 \
      --strategy all \
      --pilot

  Geração de exemplos utilizando a estratégia 'template':

    uv run python src/data/generate_synthetic.py \
      --backend gemini \
      --strategy template \
      --target 20000

Saída:
  Dataset sintético contendo exemplos destinados às classes DD e AMB,
  conforme a estratégia de geração selecionada.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from google import genai
from google.genai import types


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DATA_PENDING, Sample, append_jsonl, normalize_text

REPO_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = REPO_DIR / "prompts"
CONTEXT_PATH = PROMPTS_DIR / "context_schema.yaml"
CHECKPOINT_PATH = DATA_PENDING / "synthetic_checkpoint.json"
OUTPUT_PATH = DATA_PENDING / "dd_amb_synthetic.jsonl"

BATCH_SIZE = 8

class LLMBackend(ABC):
  name: str

  @abstractmethod
  def generate(self, prompt: str, temperature: float) -> str:
    """Retorna o texto bruto gerado pelo modelo para o prompt dado."""

class GeminiBackend(LLMBackend):
  def __init__(self, model: str | None = None):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
      raise RuntimeError("GEMINI_API_KEY não definida no ambiente (.env)")
    self.client = genai.Client(api_key=api_key)
    self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    self.name = f"gemini:{self.model}"

  @retry(wait=wait_exponential(multiplier=2, min=2, max=60), stop=stop_after_attempt(5))
  def generate(self, prompt: str, temperature: float) -> str:
    resp = self.client.models.generate_content(
      model=self.model,
      contents=prompt,
      config=types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json",
      ),
    )
    return resp.text

class MistralBackend(LLMBackend):
  def __init__(self, model: str | None = None):
    try:
      from mistralai import Mistral
    except ImportError:
      from mistralai.client import Mistral  # type: ignore

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
      raise RuntimeError("MISTRAL_API_KEY não definido no ambiente (.env)")
    self.client = Mistral(api_key=api_key)
    self.model = model or os.environ.get("MISTRAL_MODEL", "mistral-small-latest")
    self.name = f"mistral:{self.model}"

  @retry(wait=wait_exponential(multiplier=2, min=2, max=60), stop=stop_after_attempt(5))
  def generate(self, prompt:str, temperature: float) -> str:
    resp = self.client.chat.complete(
      model=self.model,
      messages=[{"role": "user", "content": prompt}],
      temperature=temperature,
    )
    return resp.choices[0].message.content


class Gemma4Backend(LLMBackend):
  """Backend local, servido via Ollama (https://ollama.com).

  Requer o servidor Ollama em execução (`ollama serve`) e o modelo já
  baixado — caso não esteja disponível localmente, é baixado
  automaticamente na primeira inicialização (`ollama pull <modelo>`).
  """

  def __init__(self, model: str | None = None):
    try:
      import ollama
    except ImportError as e:
      raise RuntimeError(
        "Pacote 'ollama' não instalado. Rode `uv add ollama` (ou `pip install ollama`)."
      ) from e

    self.model = model or os.environ.get("GEMMA_MODEL", "gemma4:e4b")
    host = os.environ.get("OLLAMA_HOST")
    self.client = ollama.Client(host=host) if host else ollama.Client()
    self.name = f"gemma4:{self.model}"

    try:
      local_models = {
        getattr(m, "model", None) or m.get("model") for m in self.client.list().models
      }
    except Exception as e:
      raise RuntimeError(
        "Não foi possível conectar ao servidor Ollama. Rode `ollama serve` "
        "e garanta que o host esteja acessível (ver variável OLLAMA_HOST)."
      ) from e

    if self.model not in local_models:
      print(f"[gemma4] modelo '{self.model}' não encontrado localmente, baixando via ollama...")
      self.client.pull(self.model)

  @retry(wait=wait_exponential(multiplier=2, min=2, max=60), stop=stop_after_attempt(5))
  def generate(self, prompt: str, temperature: float) -> str:
    # Nota: não usamos o parâmetro `format="json"` do Ollama aqui — ele força
    # um objeto JSON na raiz da resposta, o que faz o modelo "inventar" as
    # sentenças como chaves de um objeto em vez de devolver o array pedido no
    # prompt. Sem essa restrição, o modelo segue a instrução textual do
    # prompt e retorna o array corretamente.
    resp = self.client.chat(
      model=self.model,
      messages=[{"role": "user", "content": prompt}],
      options={"temperature": temperature},
    )
    return resp["message"]["content"]


BACKENDS = {"gemini": GeminiBackend, "mistral": MistralBackend, "gemma4": Gemma4Backend}


def get_backend(name: str) -> LLMBackend:
    if name not in BACKENDS:
        raise ValueError(f"Backend desconhecido: {name}. Opções: {list(BACKENDS)}")
    return BACKENDS[name]()

# --------------------------------------------------------------------------
# Parsing robusto de saída JSON do LLM
# --------------------------------------------------------------------------


def parse_json_array(raw: str) -> list:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    # Alguns modelos (sobretudo backends locais menores, como o Gemma4) em
    # modo JSON retornam um objeto envolvendo a lista (ex.: {"enunciados":
    # [...]})  em vez do array puro pedido no prompt. Tenta localizar a
    # primeira lista dentro do objeto antes de desistir.
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Não foi possível localizar um array JSON na resposta: {raw[:200]!r}") from e
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for value in obj.values():
            if isinstance(value, list):
                return value
    raise ValueError(f"Não foi possível localizar um array JSON na resposta: {raw[:200]!r}")


# --------------------------------------------------------------------------
# Checkpoint (retomabilidade)
# --------------------------------------------------------------------------


def load_checkpoint() -> set:
    if CHECKPOINT_PATH.exists():
        return set(json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8")))
    return set()


def save_checkpoint(done: set) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(json.dumps(sorted(done)), encoding="utf-8")


def job_id(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# Taxonomia e prompts
# --------------------------------------------------------------------------


def load_taxonomy() -> dict:
    return yaml.safe_load(CONTEXT_PATH.read_text(encoding="utf-8"))


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Estratégias de geração
# --------------------------------------------------------------------------


def run_template_strategy(backend: LLMBackend, taxonomy: dict, target: int, temperature: float, done: set):
    template = load_prompt("template_paraphrase")
    categories = list(taxonomy["categories"].items())
    n_written = 0
    for batch_idx in itertools.count():
        if n_written >= target:
            break
        cat_key, cat = categories[batch_idx % len(categories)]
        jid = job_id("template", backend.name, cat_key, str(batch_idx // len(categories)))
        if jid in done:
            continue
        prompt = template.format(
            category_label=cat["label"],
            examples="\n".join(f"- {e}" for e in cat["examples"]),
            n=BATCH_SIZE,
        )
        try:
            raw = backend.generate(prompt, temperature)
            items = parse_json_array(raw)
        except Exception as e:  # noqa: BLE001
            print(f"[template] falha no lote {jid} ({cat_key}): {e}")
            continue
        samples = [
            Sample(
                text=normalize_text(str(t)),
                label="DD",
                source="synthetic:template",
                strategy="template_paraphrase",
                model=backend.name,
                meta={"category": cat_key},
            )
            for t in items
            if str(t).strip()
        ]
        n_written += append_jsonl(samples, OUTPUT_PATH)
        done.add(jid)
        save_checkpoint(done)
        print(f"[template] lote {jid} ({cat_key}): +{len(samples)} (total {n_written}/{target})")
        time.sleep(1)  # respeito a rate limit de API gratuita


def run_roleplay_strategy(backend: LLMBackend, taxonomy: dict, target: int, temperature: float, done: set):
    template = load_prompt("roleplay_contextual")
    categories = list(taxonomy["categories"].items())
    contexts = taxonomy["contexts"]
    tones = taxonomy["tones"]
    combos = list(itertools.product(categories, contexts, tones))
    random.Random(42).shuffle(combos)
    n_written = 0
    for idx, ((cat_key, cat), context, tone) in enumerate(itertools.cycle(combos)):
        if n_written >= target:
            break
        jid = job_id("roleplay", backend.name, cat_key, context, tone, str(idx // len(combos)))
        if jid in done:
            continue
        prompt = template.format(
            context=context,
            tone=tone,
            category_label=cat["label"],
            examples=", ".join(cat["examples"][:3]),
            n=BATCH_SIZE,
        )
        try:
            raw = backend.generate(prompt, temperature)
            items = parse_json_array(raw)
        except Exception as e:  # noqa: BLE001
            print(f"[roleplay] falha no lote {jid}: {e}")
            continue
        samples = [
            Sample(
                text=normalize_text(str(t)),
                label="DD",
                source="synthetic:roleplay",
                strategy="roleplay_contextual",
                model=backend.name,
                meta={"category": cat_key, "context": context, "tone": tone},
            )
            for t in items
            if str(t).strip()
        ]
        n_written += append_jsonl(samples, OUTPUT_PATH)
        done.add(jid)
        save_checkpoint(done)
        print(f"[roleplay] lote {jid} ({cat_key}/{context}/{tone}): +{len(samples)} (total {n_written}/{target})")
        time.sleep(1)


def run_dialogue_strategy(backend: LLMBackend, taxonomy: dict, target: int, temperature: float, done: set):
    template = load_prompt("dialogue_synthetic")
    categories = list(taxonomy["categories"].items())
    contexts = taxonomy["contexts"]
    tones = taxonomy["tones"]
    combos = list(itertools.product(categories, contexts, tones))
    random.Random(7).shuffle(combos)
    n_written = 0
    for idx, ((cat_key, cat), context, tone) in enumerate(itertools.cycle(combos)):
        if n_written >= target:
            break
        jid = job_id("dialogue", backend.name, cat_key, context, tone, str(idx // len(combos)))
        if jid in done:
            continue
        prompt = template.format(
            context=context,
            tone=tone,
            category_label=cat["label"],
            examples=", ".join(cat["examples"][:3]),
        )
        try:
            raw = backend.generate(prompt, temperature)
            items = parse_json_array(raw)
        except Exception as e:  # noqa: BLE001
            print(f"[dialogue] falha no lote {jid}: {e}")
            continue
        samples = [
            Sample(
                text=normalize_text(str(it.get("text", ""))),
                label=it.get("label"),
                source="synthetic:dialogue",
                strategy="dialogue_synthetic",
                model=backend.name,
                meta={"category": cat_key, "context": context, "tone": tone},
            )
            for it in items
            if isinstance(it, dict)
            and it.get("label") in ("DD", "AMB")  # descarta turnos NDD do diálogo
            and str(it.get("text", "")).strip()
        ]
        n_written += append_jsonl(samples, OUTPUT_PATH)
        done.add(jid)
        save_checkpoint(done)
        print(f"[dialogue] lote {jid} ({cat_key}/{context}/{tone}): +{len(samples)} DD/AMB (total {n_written}/{target})")
        time.sleep(1)


STRATEGIES = {
    "template": run_template_strategy,
    "roleplay": run_roleplay_strategy,
    "dialogue": run_dialogue_strategy,
}


def main() -> None:
    load_dotenv(REPO_DIR / ".env")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=list(BACKENDS), required=True)
    parser.add_argument("--strategy", choices=list(STRATEGIES) + ["all"], default="all")
    parser.add_argument("--target", type=int, default=None, help="Volume-alvo por estratégia")
    parser.add_argument("--pilot", action="store_true", help="Piloto pequeno (50 amostras/estratégia) para revisão manual")
    parser.add_argument("--temperature", type=float, default=0.9)
    args = parser.parse_args()

    target = args.target or (50 if args.pilot else 1000)

    backend = get_backend(args.backend)
    taxonomy = load_taxonomy()
    done = load_checkpoint()

    strategies = list(STRATEGIES) if args.strategy == "all" else [args.strategy]
    for strat in strategies:
        print(f"=== estratégia: {strat} | backend: {backend.name} | alvo: {target} ===")
        STRATEGIES[strat](backend, taxonomy, target, args.temperature, done)

    print(f"Concluído. Amostras acumuladas em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
