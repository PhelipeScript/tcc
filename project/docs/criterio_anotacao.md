# Critério de anotação DD/NDD

Este critério reúne, num texto só, as regras usadas na construção do corpus:
- os prompts de geração e de autoverificação (`ddsd/corpus/generation/prompts.py`);
- os filtros de exclusão (`ddsd/corpus/generation/generator.py`, `config/corpus.toml`);
- a pergunta da revisão humana (`ddsd corpus annotate`).

A pergunta feita a quem anota foi uma só: **"esta frase é dirigida ao computador (DD) ou não (NDD)?"**.

## Cena de referência

Uma pessoa está perto de um computador pessoal cujo assistente de voz ouve tudo o que é dito, sem palavra de
ativação. Cada frase é julgada sozinha, sem o histórico da conversa.

## DD: fala dirigida ao computador

A frase pede ao computador uma **ação de software** que ele executa. Também é DD a pergunta que o assistente
responderia.

- **Ações de software**, nas 7 categorias do esquema (`config/context_schema.yaml`):
  - controle de aplicativos: "abre o navegador";
  - captura de tela: "tira um print da tela";
  - mídia: "pausa o vídeo";
  - sistema: "diminui o brilho da tela";
  - navegador e web: "pesquisa a previsão do tempo";
  - produtividade e arquivos: "me lembra de enviar o relatório às três";
  - ditado e texto: "copia esse trecho".
- **A forma não importa:** a frase é DD mesmo curta, informal, educada, indireta ("dá pra pausar o vídeo?")
  ou longa e cheia de detalhes ("então, abre o navegador e procura aquela receita…").

## NDD: fala não dirigida ao computador

Todo o resto:
- conversa com outra pessoa: "você vai almoçar em casa?";
- comentário ou reação: "nossa, que frio hoje";
- fala sobre o computador sem pedir nada a ele: "esse computador tá muito lento hoje";
- **pedido a uma pessoa de uma ação física ou pessoal**, que o computador não faz: "me passa o carregador",
  "fecha a porta aí". Isso vale mesmo no imperativo.

## Regra de desempate

O que decide é **a ação pedida**, e não a forma da frase.
- Ação de software: DD, mesmo com jeito de conversa ("bota uma música aí").
- Ação física ou pessoal: NDD, mesmo no imperativo ("abre a janela aí", que é a janela da sala).

Os dois prompts de autoverificação tratam a dúvida de formas diferentes:
- o da estratégia de comprimento manda responder DD na dúvida, se a frase sozinha **poderia** ser um comando
  ao computador;
- o dos negativos difíceis classifica só pela ação pedida.

## Exclusões

Ficam fora do corpus:
- frases com **palavra de ativação**: "alexa", "siri", "ok google", "hey google" e "assistente". O sistema
  estudado funciona sem ela;
- frases de **casa inteligente** (luz, ar-condicionado, cafeteira, ventilador etc.), porque o domínio é o
  computador pessoal;
- frases com até 5 caracteres, com menos de 2 palavras ou com mais de 50 palavras;
- frases com caracteres fora do ASCII depois da normalização;
- listas serializadas: um defeito de gerador encontrado no corpus inicial;
- duplicatas exatas e quase-duplicatas (cosseno TF-IDF > 0,95);
- **textos que aparecem nas duas classes**, que saem das duas.
