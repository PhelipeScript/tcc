"""
Geração sintética das classes DD e NDD via API Mistral para o corpus DDSD.

Estratégias:
  1. DD — paráfrase transitória (template): mesma intenção, palavras diferentes.
  2. DD — roleplay contextual: personas (idoso formal, jovem/gamer).
  3. NDD — conversa fiada/fundo: mesmas palavras-chave em contexto conversacional.

Saída (formato final do usuário: apenas text e label):
  project/mistral/dd.jsonl   -> {"text": ..., "label": "DD"}
  project/mistral/ndd.jsonl  -> {"text": ..., "label": "NDD"}

Uso:
  cd project
  uv run python src/dataset/mistral_generate.py --target 250
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import normalize_text  # noqa: E402

REPO_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = REPO_DIR / "prompts"
CONTEXT_PATH = PROMPTS_DIR / "context_schema.yaml"
PROJECT_ROOT = REPO_DIR.parent.parent
OUT_DIR = REPO_DIR / "mistral"
OUT_DD = OUT_DIR / "dd.jsonl"
OUT_NDD = OUT_DIR / "ndd.jsonl"

WAKE_WORDS = ("alexa", "siri", "ok google", "ok, google", "assistente", "hey google")

DD_TEMPLATE = """Você é um gerador de dados para um corpus de comandos de voz em português brasileiro.

Categoria de comando: {category_label}
Exemplos de referência desta categoria (não repita literalmente, use como inspiração):
{examples}

Gere {n} enunciados NOVOS e NATURAIS em português brasileiro que uma pessoa diria em voz alta para
executar uma ação da categoria acima em um computador pessoal. Cada um é um COMANDO CLARAMENTE
DIRECIONADO ao dispositivo (classe DD).

Varie: formalidade (informal a educado), estrutura sintática (imperativo direto, pergunta indireta,
pedido com "pode"/"consegue"/"dá pra"), e marcadores de fala coloquial ("né", "tipo", "então", "por favor").
Se o comando de base é "aumentar volume", gere paráfrases como "bota mais alto", "dá um gás no som",
"está muito baixo, aumenta" — foque na semântica, não nas palavras exatas.

Regras:
- NÃO inclua palavra de ativação (Alexa, Siri, Ok Google, Assistente).
- NÃO repita os exemplos literalmente.
- Enunciado curto (3 a 20 palavras), como fala real.

Responda APENAS com um array JSON de strings, sem texto adicional, sem markdown:
["enunciado 1", "enunciado 2", ...]"""

DD_ROLEPLAY = """Você é um gerador de dados para um corpus de comandos de voz em português brasileiro.

Assuma o papel de um(a) usuário(a) de computador pessoal com este perfil:
- Persona: {persona}  (ex.: idoso que fala de forma formal e educada, jovem/gamer com gírias e frases
  truncadas, criança pedindo de forma simples, profissional apressado, pessoa sussurrando num ambiente
  compartilhado, pessoa irritada)
- Contexto de uso do computador: {context}
- Categoria de ação que quer realizar: {category_label}

Gere {n} enunciados NOVOS que essa pessoa diria em voz alta ao assistente de voz do computador para
executar a ação da categoria, coerentes com a persona e o contexto. São comandos CLARAMENTE
DIRECIONADOS ao dispositivo (classe DD).

Regras:
- Reflita a persona (ex.: idoso: "Por gentileza, você poderia fechar esta janela?"; jovem: "Muta aí, muta aí").
- NÃO inclua palavra de ativação (Alexa, Siri, Ok Google, Assistente).
- NÃO repita os exemplos literalmente. Enunciado curto (3 a 20 palavras).

Responda APENAS com um array JSON de strings, sem texto adicional, sem markdown:
["enunciado 1", "enunciado 2", ...]"""

NDD_TEMPLATE = """Você é um gerador de dados para um corpus de fala em português brasileiro.

Aqui está um comando típico que uma pessoa dá a um assistente de voz (CLASSE DD / comando):
  comando: "{dd_text}"

Tarefa: gere {n} frases DIFERENTES de CONVERSA FIADA / FALA DE FUNDO (CLASSE NDD) em português
brasileiro natural e espontâneo, em que uma pessoa FALA O MESMO TEMA/vocabulário do comando, mas em
situação conversacional, dirigindo-se a outro humano ou comentando em voz alta — SEM intenção de
comandar o dispositivo.

Exemplos do padrão desejado:
  comando "liga a TV"        -> "ontem eu liguei a TV e só passava jornal"
  comando "que horas são?"   -> "não sei que horas são o jogo"
  comando "tá frio aqui"     -> "o café tá frio"
  comando "aumenta o volume" -> "o volume aqui é baixo demais pro show de ontem"

Regras:
- Use o mesmo tema/vocabulário, mas em contexto de conversa (humano-humano ou comentário incidental).
- NÃO soe como ordem direta ao dispositivo. NÃO inclua palavra de ativação (Alexa, Siri, Assistente).
- NÃO repita literalmente o comando nem os exemplos. Fala natural (5 a 25 palavras).

Responda APENAS com um array JSON de strings, sem texto adicional, sem markdown:
["frase 1", "frase 2", ...]"""


class MistralClient:
    def __init__(self, model: str | None = None):
        try:
            from mistralai import Mistral
        except ImportError:
            from mistralai.client import Mistral  # type: ignore
        import os

        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY não definida (cheque project/.env)")
        self.client = Mistral(api_key=api_key)
        self.model = model or os.environ.get("MISTRAL_MODEL", "mistral-small-latest")

    @retry(wait=wait_exponential(multiplier=3, min=5, max=120), stop=stop_after_attempt(10))
    def generate(self, prompt: str, temperature: float) -> str:
        resp = self.client.chat.complete(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return resp.choices[0].message.content


def parse_json_array(raw: str) -> list:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON não localizado: {raw[:200]!r}") from e
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, list):
                return v
    raise ValueError(f"JSON não localizado: {raw[:200]!r}")


def clean(text: str, label: str) -> str | None:
    text = normalize_text(text)
    if not text:
        return None
    low = text.lower()
    if any(w in low for w in WAKE_WORDS):
        return None
    return text


def read_jsonl_texts(path: Path) -> set:
    if not path.exists():
        return set()
    out = set()
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        out.add(json.loads(line)["text"])
    return out


def append_text(path: Path, texts: list[str], label: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for t in texts:
            c = clean(t, label)
            if not c:
                continue
            f.write(json.dumps({"text": c, "label": label}, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_taxonomy() -> dict:
    return yaml.safe_load(CONTEXT_PATH.read_text(encoding="utf-8"))


PERSONAS = [
    "idoso(a), que fala de forma formal, educada e pausada (ex.: 'Por gentileza, você poderia fechar esta janela?')",
    "jovem/gamer, que usa gírias e frases truncadas (ex.: 'Muta aí, muta aí', 'Bota mais alto')",
    "profissional apressado, frases curtas e diretas",
    "criança, pedindo de forma simples e espontânea",
    "pessoa sussurrando em ambiente compartilhado, tom bem baixo",
    "pessoa irritada/frustrada, leve impaciência",
]


def generate_dd_template(client, cat, n, existed):
    prompt = DD_TEMPLATE.format(
        category_label=cat["label"],
        examples="\n".join(f"- {e}" for e in cat["examples"]),
        n=n,
    )
    items = [clean(str(i), "DD") for i in parse_json_array(client.generate(prompt, 0.9))]
    items = [i for i in items if i and i not in existed]
    return items


def generate_dd_roleplay(client, cat, context, persona, n, existed):
    prompt = DD_ROLEPLAY.format(
        persona=persona,
        context=context,
        category_label=cat["label"],
        n=n,
    )
    items = [clean(str(i), "DD") for i in parse_json_array(client.generate(prompt, 0.9))]
    items = [i for i in items if i and i not in existed]
    return items


def generate_ndd(client, dd_text, n, existed):
    prompt = NDD_TEMPLATE.format(dd_text=dd_text, n=n)
    items = [clean(str(i), "NDD") for i in parse_json_array(client.generate(prompt, 0.9))]
    items = [i for i in items if i and i not in existed]
    return items


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=250, help="Amostras-alvo por classe")
    parser.add_argument("--seed-ndd", type=str, default=None,
                        help="Arquivo dd.jsonl (manual) cujas frases guiam a geração NDD")
    args = parser.parse_args()

    client = MistralClient()
    taxonomy = load_taxonomy()
    categories = list(taxonomy["categories"].items())
    contexts = taxonomy["contexts"]
    rng = random.Random(42)

    dd_existed = read_jsonl_texts(OUT_DD)
    dd_seed = set()
    if args.seed_ndd:
        dd_seed = read_jsonl_texts(Path(args.seed_ndd))

    BATCH = 8
    RATE_SLEEP = 12
    print(f"Mistral {client.model} | alvo DD e NDD: {args.target}")

    # ---- DD ----
    n_dd = len(dd_existed)
    combos = [(cat, ctx, p) for cat in categories for ctx in contexts for p in PERSONAS]
    rng.shuffle(combos)
    ci = 0
    while n_dd < args.target:
        cat, ctx, persona = combos[ci % len(combos)]
        try:
            items = generate_dd_roleplay(client, cat[1], ctx, persona, BATCH, dd_existed)
        except Exception as e:  # noqa: BLE001
            print(f"[dd:roleplay] falha no lote: {e}", flush=True)
            ci += 1
            time.sleep(RATE_SLEEP)
            continue
        got = append_text(OUT_DD, items, "DD")
        dd_existed.update(items)
        n_dd += got
        ci += 1
        print(f"[dd:roleplay] +{got} (total {n_dd}/{args.target})", flush=True)
        time.sleep(RATE_SLEEP)
        # intercala template/paráfrase
        if ci % 2 == 0 and n_dd < args.target:
            cat2 = categories[ci % len(categories)][1]
            try:
                items = generate_dd_template(client, cat2, BATCH, dd_existed)
            except Exception as e:  # noqa: BLE001
                print(f"[dd:template] falha no lote: {e}", flush=True)
                time.sleep(RATE_SLEEP)
                continue
            got = append_text(OUT_DD, items, "DD")
            dd_existed.update(items)
            n_dd += got
            print(f"[dd:template] +{got} (total {n_dd}/{args.target})", flush=True)
            time.sleep(RATE_SLEEP)

    # ---- NDD ----
    ndd_existed = read_jsonl_texts(OUT_NDD)
    n_ndd = len(ndd_existed)
    seeds = list(dd_seed or dd_existed)
    if not seeds:
        seeds = [c[1]["examples"][0] for c in categories]
    si = 0
    while n_ndd < args.target:
        seed = seeds[si % len(seeds)]
        try:
            items = generate_ndd(client, seed, BATCH, ndd_existed)
        except Exception as e:  # noqa: BLE001
            print(f"[ndd] falha no lote: {e}", flush=True)
            si += 1
            time.sleep(RATE_SLEEP)
            continue
        got = append_text(OUT_NDD, items, "NDD")
        ndd_existed.update(items)
        n_ndd += got
        si += 1
        print(f"[ndd] +{got} (total {n_ndd}/{args.target})", flush=True)
        time.sleep(RATE_SLEEP)

    print(f"Feito. DD em {OUT_DD} ({n_dd}), NDD em {OUT_NDD} ({n_ndd})")


if __name__ == "__main__":
    main()
