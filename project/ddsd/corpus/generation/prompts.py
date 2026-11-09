"""Prompts da estratégia `comprimento` (mitigação do viés curto = DD / longo = NDD), versão 2.

Há dois prompts de geração, um por classe, parametrizados pelo número exato de
palavras `k`, e um prompt de autoverificação, no qual o mesmo modelo
reclassifica cada frase.

O piloto mostrou dois problemas, que esta versão resolve:
1. Eco dos exemplos. Com exemplos fixos, o modelo os repetia e parafraseava
   ("você viu meu fone", "planilha de vendas do terceiro trimestre"). Agora cada
   lote sorteia poucos exemplos de bancos maiores e foca em 2-3 tipos de fala,
   o que diversifica o conjunto.
2. Comando disfarçado de NDD. Dar exemplos de comandos como dica de
   vocabulário no prompt de NDD induzia imperativos. No NDD vai só o tema.

A autoverificação descarta o que sobrar de rótulo errado. Com o raciocínio
desligado, modelos pequenos deixam passar "abre outra janela" como NDD.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass

PROMPT_VERSION = "comprimento-v2"

#: Tipos de fala NDD e um banco de exemplos de cada um.
NDD_TYPES: dict[str, tuple[str, ...]] = {
    "reação ou comentário em voz alta": (
        "travou de novo", "nossa, que lentidão", "isso tá bugado hoje", "olha que estranho",
        "caramba, sumiu tudo", "ficou bonito esse gráfico", "esse site é muito lento"),
    "resposta a alguém da sala": (
        "já salvei, pode deixar", "tá bom, depois eu vejo", "não, ainda não mandei", "pode ser, tanto faz",
        "acho que foi ontem", "sim, tá aberto aqui"),
    "pergunta a uma pessoa": (
        "cê já almoçou?", "quem mexeu aqui?", "você mandou o arquivo?", "a reunião é que horas mesmo?",
        "você tá ouvindo esse barulho?", "cadê o pessoal?"),
    "pedido a uma pessoa presente, sobre algo físico ou pessoal": (
        "segura isso pra mim", "fala mais baixo, por favor", "me empresta tua caneta", "chega mais perto aqui",
        "fecha a porta aí, tá frio", "pega uma água pra mim?"),
    "fala consigo mesma": (
        "deixa eu pensar aqui", "esqueci a senha de novo", "agora vai", "calma, quase lá",
        "onde foi parar aquilo", "acho que errei alguma coisa"),
    "trecho de conversa ao telefone": (
        "oi, pode falar", "te ligo daqui a pouco", "não, tô no computador ainda", "tá, já tô indo",
        "manda no zap depois", "tô terminando um negócio aqui"),
}

#: Estilos de comando DD e um banco de exemplos de cada um.
DD_STYLES: dict[str, tuple[str, ...]] = {
    "imperativo direto": ("fecha essa aba", "pausa o vídeo", "abre o chrome", "tira um print", "pula essa música"),
    "pedido indireto": ("dá pra pausar o vídeo?", "consegue fechar essa janela?", "tem como aumentar o brilho?",
                        "você pode abrir o spotify?"),
    "pergunta que o assistente responde": ("qual música tá tocando?", "quanto falta pro download acabar?",
                                           "qual é a última aba que eu fechei?"),
}

#: Recursos para chegar a comandos longos de forma natural, com exemplos rotativos.
LONG_STRATEGIES: dict[str, tuple[str, ...]] = {
    "polidez": ("você pode, por favor", "se não for incômodo", "faz um favor pra mim"),
    "detalhe do alvo": ("o vídeo que eu tava vendo ontem à noite", "a pasta de fotos da viagem",
                        "aquele documento com as notas da reunião"),
    "justificativa": ("que eu vou apresentar daqui a pouco", "porque a reunião começa em cinco minutos",
                      "que o professor pediu pra hoje"),
    "duas ações encadeadas": ("salva o documento e depois fecha o editor", "abre o navegador e entra no meu e-mail"),
    "hesitação ou autocorreção": ("é...", "deixa eu ver", "não, espera, a outra"),
}

_COMMON_RULES = """Regras gerais:
- Português brasileiro falado, natural e espontâneo (pode usar "né", "tipo", "então", "pô", "ué", "tá", "pra").
- Sem palavra de ativação (Alexa, Siri, Ok Google, Assistente) e sem casa inteligente ou aparelhos físicos
  controlados por voz (luz, lâmpada, cortina, ar-condicionado, ventilador, cafeteira etc.).
- Todas diferentes entre si: varie o começo, a estrutura e o vocabulário. Não use nem parafraseie os exemplos
  acima, que só ilustram o estilo.
- Tamanho: EXATAMENTE {k} palavras em cada frase. Conte antes de responder ("tá" e "pra" contam como uma
  palavra cada)."""

NDD_PROMPT = """Você cria dados para uma pesquisa sobre assistentes de voz em computadores pessoais, em português
brasileiro. O sistema estudado ouve tudo o que é dito perto do computador e precisa ignorar o que NÃO é
dirigido a ele.

Cena: a pessoa está {context}, perto do computador. Ela fala em tom {tone}. Assunto que pode aparecer:
{category}.

Escreva {n} falas DIFERENTES que essa pessoa diria em voz alta, mas que NÃO são dirigidas ao computador
(classe NDD), com EXATAMENTE {k} palavras cada.

Neste lote, use principalmente estes tipos de fala (exemplos só de estilo):
{types}

PROIBIDO, porque seria entendido como comando ao computador:
- verbo de ação no imperativo que o computador executaria: abrir, fechar, tocar, pausar, pesquisar, salvar,
  copiar, aumentar ou diminuir o volume etc.;
- pergunta que o assistente responderia (as horas, o clima, o que está tocando).
O vocabulário do computador pode aparecer, mas só para descrever, comentar ou perguntar a uma pessoa.

{rules}

Responda APENAS com um array JSON de {n} strings, sem markdown e sem comentários:
["fala 1", "fala 2", ...]"""

DD_PROMPT = """Você cria dados para uma pesquisa sobre assistentes de voz em computadores pessoais, em português
brasileiro. O sistema estudado ouve tudo o que é dito perto do computador e precisa reconhecer quando a
pessoa está pedindo algo a ele.

Cena: a pessoa está {context}, usando o computador. Ela fala em tom {tone}. Ação desejada: {category}
(ações típicas: {examples}).

Escreva {n} comandos DIFERENTES que essa pessoa diria em voz alta ao assistente de voz do computador para
executar uma ação dessa categoria (classe DD), com EXATAMENTE {k} palavras cada.

Estilos deste lote (exemplos só de estilo):
{styles}
{size_hint}

Precisa ficar claro, só pela frase, que é um pedido ao computador para executar uma ação de SOFTWARE.

{rules}

Responda APENAS com um array JSON de {n} strings, sem markdown e sem comentários:
["comando 1", "comando 2", ...]"""

_SHORT_HINT = "Com poucas palavras, seja direto como uma ordem rápida."

JUDGE_PROMPT = """Você revisa frases de um corpus de pesquisa sobre assistentes de voz de computador pessoal.
Imagine cada frase dita em voz alta perto de um computador cujo assistente de voz está sempre ouvindo e
classifique:
- "DD": pedido para o computador executar algo (abrir, fechar, tocar, pausar, pesquisar, salvar, escrever,
  ajustar volume ou brilho etc.) ou pergunta que o assistente responderia, mesmo curto, informal ou
  indireto;
- "NDD": não é para o computador: conversa ou pergunta a outra pessoa, comentário, reação, pedido a uma
  pessoa sobre algo físico ou pessoal.
Na dúvida, isto é, se a frase sozinha PODERIA ser um comando ao computador, responda "DD".

Frases (JSON):
{sentences}

Responda APENAS com um array JSON de {n} rótulos ("DD" ou "NDD"), na mesma ordem das frases."""


@dataclass(frozen=True, slots=True)
class PromptSpec:
    """Parâmetros de um lote: classe, tamanho e a combinação da taxonomia."""

    label: str
    words: int
    n: int
    category: str
    examples: tuple[str, ...]
    context: str
    tone: str


def _bullets(pool: dict[str, tuple[str, ...]], rng: random.Random, kinds: int) -> str:
    chosen = rng.sample(sorted(pool), k=min(kinds, len(pool)))
    return "\n".join(f'- {kind} (ex.: "{rng.choice(pool[kind])}")' for kind in chosen)


def render_prompt(spec: PromptSpec, rng: random.Random, short_max_words: int = 6) -> str:
    """Monta o prompt de geração do lote, sorteando foco e exemplos com `rng`."""
    rules = _COMMON_RULES.format(k=spec.words)
    common = {"n": spec.n, "k": spec.words, "context": spec.context, "tone": spec.tone,
              "category": spec.category, "rules": rules}
    if spec.label == "NDD":
        return NDD_PROMPT.format(types=_bullets(NDD_TYPES, rng, kinds=3), **common)
    if spec.words <= short_max_words:
        hint = _SHORT_HINT
    else:
        hint = ("Para chegar ao tamanho de forma natural, combine recursos como:\n"
                + _bullets(LONG_STRATEGIES, rng, kinds=3))
    examples = ", ".join(f'"{e}"' for e in rng.sample(list(spec.examples), k=min(3, len(spec.examples))))
    return DD_PROMPT.format(styles=_bullets(DD_STYLES, rng, kinds=2), size_hint=hint, examples=examples, **common)


def render_judge(sentences: Sequence[str]) -> str:
    """Monta o prompt de autoverificação de um lote de frases."""
    return JUDGE_PROMPT.format(sentences=json.dumps(list(sentences), ensure_ascii=False), n=len(sentences))


# --------------------------------------------------------------------------------------------
# Estratégia `negativos_dificeis`: mesma forma superficial (pedido/imperativo curto), classes
# separadas só pela AÇÃO pedida - física/pessoal a uma pessoa (NDD) × de software ao computador (DD).
# Cobre a lacuna medida no corpus: pedidos a pessoas eram 3,1% do NDD curto do corpus inicial e o
# juiz conservador da estratégia `comprimento` descartava justamente esses casos.
# --------------------------------------------------------------------------------------------
HARD_PROMPT_VERSION = "negativos_dificeis-v1"

HARD_NDD_EXAMPLES = (
    "me passa o carregador", "segura isso aqui", "espera um pouco", "vem cá rapidinho", "fecha a porta aí",
    "pega uma água pra mim", "fala mais baixo", "senta aqui do lado", "vai em frente", "me empresta a caneta",
    "chama o Pedro pra mim", "traz o caderno", "olha isso aqui", "joga o lixo fora", "abre a janela aí",
)
HARD_DD_EXAMPLES = (
    "me mostra a agenda de hoje", "bota uma música aí", "abre o spotify", "manda esse arquivo por e-mail",
    "fecha essa aba aí", "coloca em tela cheia", "me lembra da reunião", "pesquisa o horário do jogo",
    "salva isso aí", "volta o vídeo um pouco", "aumenta o volume aí", "copia esse texto",
)

HARD_NDD_PROMPT = """Você cria dados para uma pesquisa sobre assistentes de voz em computadores pessoais, em português
brasileiro. O sistema estudado ouve tudo o que é dito perto do computador e precisa ignorar o que NÃO é
dirigido a ele, inclusive ORDENS E PEDIDOS feitos a outras pessoas.

Cena: a pessoa está {context}, perto do computador, com outras pessoas na sala. Ela fala em tom {tone}.

Escreva {n} pedidos ou ordens DIFERENTES que ela diria a uma PESSOA presente (classe NDD), com EXATAMENTE
{k} palavras cada. A ação pedida tem de ser FÍSICA ou PESSOAL, algo que um computador não consegue fazer:
passar ou trazer um objeto, abrir ou fechar a porta ou a janela, esperar, vir até ali, falar mais baixo,
sentar, olhar algo, chamar alguém, ajudar com alguma coisa, ir em frente com uma tarefa etc.
Exemplos só de estilo (não copie): {examples}

Use formas imperativas e pedidos diretos, como numa conversa real ("me passa", "pega", "vem", "espera",
"segura", "traz", "pode me ajudar", "vai lá"). NÃO peça ações de software (abrir programa, tocar música,
pesquisar, salvar arquivo etc.).

{rules}

Responda APENAS com um array JSON de {n} strings, sem markdown e sem comentários:
["pedido 1", "pedido 2", ...]"""

HARD_DD_PROMPT = """Você cria dados para uma pesquisa sobre assistentes de voz em computadores pessoais, em português
brasileiro. O sistema estudado precisa reconhecer quando a pessoa pede algo ao assistente do computador.

Cena: a pessoa está {context}, usando o computador. Ela fala em tom {tone}. Ação desejada: {category}.

Escreva {n} pedidos DIFERENTES ao assistente do computador (classe DD), com EXATAMENTE {k} palavras cada,
usando as MESMAS formas de uma conversa entre pessoas ("me mostra", "bota", "manda", "abre aí", "pega",
"pode", "dá pra", "vai", "volta"), mas pedindo uma ação de SOFTWARE que o computador executa.
Exemplos só de estilo (não copie): {examples}

{rules}

Responda APENAS com um array JSON de {n} strings, sem markdown e sem comentários:
["pedido 1", "pedido 2", ...]"""

HARD_JUDGE_PROMPT = """Você revisa frases de um corpus de pesquisa sobre assistentes de voz de computador pessoal.
Cada frase foi dita em voz alta perto de um computador cujo assistente de voz está sempre ouvindo.
Classifique PELA AÇÃO pedida:
- "DD": pede ao computador uma ação de SOFTWARE (abrir/fechar programa ou aba, tocar/pausar mídia, pesquisar,
  salvar, mostrar, mandar por e-mail/app, ajustar volume ou brilho etc.);
- "NDD": pede a uma PESSOA uma ação física ou pessoal que o computador não executa (passar/trazer objeto,
  abrir a porta, esperar, vir, falar baixo, sentar, ajudar etc.), ou é conversa/comentário.
Se a ação for de software, é DD mesmo com jeito de conversa; se for física ou pessoal, é NDD mesmo no imperativo.

Frases (JSON):
{sentences}

Responda APENAS com um array JSON de {n} rótulos ("DD" ou "NDD"), na mesma ordem das frases."""


def render_hard_prompt(spec: PromptSpec, rng: random.Random) -> str:
    """Prompt da estratégia `negativos_dificeis` (NDD: pedido físico a pessoa; DD: mesma forma, software)."""
    rules = _COMMON_RULES.format(k=spec.words)
    pool = HARD_NDD_EXAMPLES if spec.label == "NDD" else HARD_DD_EXAMPLES
    examples = ", ".join(f'"{e}"' for e in rng.sample(pool, k=4))
    template = HARD_NDD_PROMPT if spec.label == "NDD" else HARD_DD_PROMPT
    return template.format(n=spec.n, k=spec.words, context=spec.context, tone=spec.tone, category=spec.category,
                           examples=examples, rules=rules)


def render_hard_judge(sentences: Sequence[str]) -> str:
    return HARD_JUDGE_PROMPT.format(sentences=json.dumps(list(sentences), ensure_ascii=False), n=len(sentences))
