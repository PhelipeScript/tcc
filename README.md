# Detecção de Intenção Implícita para Ativação de Assistentes de Voz em Ambientes Controlados

**Trabalho de Conclusão de Curso II** — Bacharelado em Ciência da Computação
SENAC Santo Amaro · São Paulo · 2026

**Autor:** Phelipe Pereira de Souza
**Orientador:** Prof. Thyago Conchado Quintas

> **Entrega parcial — atualizado em 16/09/2026.** Este documento é atualizado conforme o trabalho avança.

---

## 1. O problema

Assistentes de voz convencionais exigem uma *wake-word* ("Alexa", "Ok Google") para
saber que estão sendo chamados. Este trabalho investiga a ativação por **intenção
implícita**: decidir, a partir apenas do enunciado, se a pessoa está falando **com
o computador** ou **perto dele**.

A tarefa é uma classificação binária entre duas classes que compartilham o mesmo
vocabulário — é justamente essa sobreposição que a torna difícil:

| Classe | Definição | Exemplo |
|---|---|---|
| **DD** (*device-directed*) | Comando dirigido ao computador | `"fecha essa aba por favor"` |
| **NDD** (*non-device-directed*) | Conversa de fundo, mesmo vocabulário | `"eu fechei sem querer a aba que tinha a pesquisa toda"` |

---

## 2. Situação em relação ao cronograma

O cronograma do TCC1 prevê **6 fases ao longo de 16 semanas**. O TCC2 teve início no
começo de agosto de 2026, o que situa esta entrega por volta da **semana 7 de 16**.

| Fase | Semanas previstas | Situação |
|---|---|---|
| **F1 — Corpus** | 1–5 | ✅ **Concluída** — corpus de 237.468 enunciados gerado e validado |
| **F2 — Implementação** | 4–9 | 🔶 **Parcial** (ver detalhe abaixo) |
| ├ Módulo ASR (Whisper) | 4–6 | ⬜ Não iniciado — **em atraso**; será feito como pipeline simulado TTS→Whisper (sem áudio real disponível), viabilidade técnica já confirmada |
| ├ Baselines clássicos (TF-IDF + SVM/LR) | 5–7 | ✅ **Concluído** (Seção 5.6) |
| └ Fine-tuning BERTimbau | 6–9 | ✅ **Concluído** |
| **F3 — Experimentos** | 8–13 | 🔵 **Adiantada** |
| └ Modelos alternativos (Albertina, DeBERTinha) | 11–13 | ✅ **Concluído, ~5 semanas adiantado** |
| **F4 — Análise** | 12–14 | 🔶 **Parcial** — overfitting de calibração e significância estatística concluídos (Seções 5.5–5.6); validação externa ainda pendente |
| **F5 — Escrita** | 1–15 | 🔵 Contínua |
| **F6 — Entrega** | 15–16 | ⬜ Não iniciado |

**Leitura honesta do quadro:** a construção do corpus foi concluída no prazo e a parte
de Transformers está adiantada — os três modelos (BERTimbau, Albertina e DeBERTinha)
já foram treinados e avaliados no conjunto de teste (Seção 5.4), sendo que os dois
alternativos estavam previstos apenas para as semanas 11–13 e eram condicionados a
"caso o cronograma permita". Em contrapartida, **duas atividades da F2 não foram
iniciadas**: o módulo ASR com Whisper (previsto para as semanas 4–6) e os baselines
clássicos (5–7). São as duas prioridades imediatas, detalhadas na Seção 6.

---

## 3. Etapa 1 — Geração do corpus com LLMs

Não existe corpus público de português brasileiro anotado em DD/NDD para o domínio de
computador pessoal. O corpus foi construído sinteticamente, com **9 fontes geradoras
independentes**, para evitar que o classificador aprenda o estilo de um único modelo
em vez da distinção semântica.

### 3.1 Modelos utilizados

**APIs gratuitas (4):**

| Fonte | Modelo |
|---|---|
| Groq | `qwen/qwen3.8-27b` |
| Mistral AI | `mistral-small-latest` |
| NVIDIA NIM | `moonshotai/kimi-k3` |
| OpenRouter | `openrouter/free` |

**Modelos locais (3),** servidos via LM Studio em máquinas próprias — um em `localhost`
e dois em servidores na rede local:

| Fonte | Modelo |
|---|---|
| LM Studio (local) | `google/gemma-4-e4b` |
| LM Studio (rede) | `qwen/qwen3.6-35b-a3b` |
| LM Studio (rede) | `zai-org/glm-4.6v-flash` |

**Agentes de código (2),** em sessões interativas de CLI em vez de geração em lote:

| Fonte | Ferramenta |
|---|---|
| `opencode` | CLI OpenCode |
| `sonnet` | Claude Sonnet |

O uso de APIs gratuitas e de inferência local foi uma decisão de projeto: manteve o
**custo de geração em zero** e permitiu gerar um volume muito acima do mínimo previsto
na metodologia (40.000 amostras), sem depender de crédito pago.

### 3.2 Estratégias de geração

Cada gerador usa três estratégias de prompt sobre uma taxonomia compartilhada
(`src/dataset/prompts/context_schema.yaml`), com **7 categorias** de comando
(controle de aplicativos, captura de tela, mídia, sistema, navegador, produtividade
e arquivos, ditado), **8 contextos** de uso e **6 tons**:

1. **Paráfrase** — variações de comandos-semente dentro de uma categoria
2. **Roleplay** — geração sob **6 personas** (idoso formal, jovem/gamer, profissional
   apressado, criança, pessoa sussurrando, pessoa irritada), para cobrir registro e
   variação sociolinguística
3. **Contraste NDD** — conversa fiada usando **o mesmo vocabulário** da classe DD, que
   é o que torna o par difícil de separar

Todos os geradores aplicam, ainda na geração, dois filtros de escopo: remoção de
*wake-words* (Alexa, Siri, Ok Google) e de comandos de casa inteligente, mantendo o
domínio estritamente em computador pessoal.

### 3.3 Volume por fonte

Snapshot de 15/09/2026, usado na reexecução do pipeline que gerou o corpus e os
modelos atuais (Seção 5):

| Fonte | DD | NDD | Total |
|---|---|---|---|
| qwen | 40.014 | 40.016 | 80.030 |
| glm | 15.082 | 15.172 | 30.254 |
| groq | 15.009 | 14.513 | 29.522 |
| lmstudio | 15.004 | 15.003 | 30.007 |
| sonnet | 15.000 | 15.000 | 30.000 |
| openrouter | 11.364 | 9.133 | 20.497 |
| nvidia | 10.002 | 9.057 | 19.059 |
| opencode | 6.067 | 6.152 | 12.219 |
| mistral | 4.974 | 5.002 | 9.976 |
| **Total** | **132.516** | **129.048** | **261.564** |

Este é o volume bruto por trás do funil da Seção 4.3; a geração continua e novos
lotes serão incorporados em execuções futuras do pipeline.

---

## 4. Etapa 2 — Preparação do dataset

Pipeline de **7 estágios**, em `project/src/data_prep/`. Cada estágio é um módulo
executável isolado, lê a saída do anterior e grava a sua, de modo que qualquer etapa
pode ser reprocessada sem refazer as anteriores. As pastas de origem são tratadas como
**somente leitura**.

```bash
uv run python -m src.data_prep.run_pipeline
```

### 4.1 Os estágios

| # | Estágio | Função |
|---|---|---|
| 1 | Snapshot | Copia as 9 fontes e grava manifesto com contagem, mtime e SHA-256 de cada arquivo |
| 2 | Normalização | Padroniza o texto e aplica filtros de integridade |
| 3 | Dedup por arquivo | Remove repetições dentro de cada arquivo |
| 4 | Merge | Consolida as 9 fontes em 2 arquivos, um por classe |
| 5 | Dedup global | Dedup entre fontes + remoção de conflitos de rótulo |
| 6 | Balanceamento | Undersample proporcional até 1:1 |
| 7 | Split | Partição 80/10/10 estratificada por classe **e** fonte |

### 4.2 Normalização

Ordem fixa: **NFKD** → remoção de marcas de acento → minúsculas → remoção de toda
pontuação e símbolo → colapso de espaços.

```
"Fecha essa aba, por favor!"   →  "fecha essa aba por favor"
"não sei que horas são"        →  "nao sei que horas sao"
"planilha do 3º trimestre"     →  "planilha do 3o trimestre"
```

A normalização é agressiva por decisão de projeto: o sistema final recebe a saída de um
ASR, que não produz pontuação nem caixa confiáveis. Normalizar treino, validação e teste
da mesma forma mantém a coerência e aproxima o texto da entrada real. O texto original é
preservado em um campo paralelo (`text_raw`), o que permite a ablação "com acento vs sem
acento" sem reprocessar nada.

### 4.3 Funil de contagens

| Estágio | DD | NDD | Observação |
|---|---|---|---|
| 1. Snapshot | 132.516 | 129.048 | corpus bruto |
| 2. Normalização | 132.462 | 129.012 | −90 (39 por tamanho, 51 por script estrangeiro) |
| 3. Dedup por arquivo | 124.642 | 128.682 | −7.820 DD, −330 NDD |
| 5. Dedup global + conflitos | 118.734 | 128.634 | −5.908 DD, −48 NDD; **43 conflitos de rótulo** |
| 6. Balanceamento 1:1 | **118.734** | **118.734** | −9.900 da classe majoritária (NDD) |
| 7. Split | 189.974 / 23.747 / 23.747 | | treino / validação / teste |

**Dataset final: 237.468 enunciados**, perfeitamente balanceado — 5,9× o volume mínimo
de 40.000 previsto na metodologia.

### 4.4 Achados relevantes para a análise

Três resultados do pipeline que valem discussão no texto do TCC:

1. **A duplicação se concentra quase toda na classe DD.** DD perde 13.728 registros
   entre os estágios 2 e 5; NDD perde apenas 378. Comandos imperativos gerados por LLM
   têm variedade lexical muito menor que conversa fiada. O desbalanceamento de 1,08:1
   antes do balanceamento é consequência disso, e não uma escolha de amostragem.

2. **43 enunciados receberam rótulos contraditórios** entre fontes — por exemplo
   `"fecha essa aba por favor"`, gerado como DD por seis fontes (glm, groq, lmstudio,
   nvidia, openrouter, qwen) e como NDD pela qwen. São casos genuinamente ambíguos,
   correspondentes à classe AMB da taxonomia. Foram removidos de **ambas** as classes
   e registrados em `conflicts.jsonl`, servindo como material qualitativo sobre a
   fronteira DD/NDD.

3. **51 enunciados continham script estrangeiro vazado pelos geradores** — chinês e
   outros idiomas dentro de frases em português, como `"juntamente com o colega调试
   系统时，nós notamos que a configuração do agregador de logs talvez precise ser
   atualizada"`. São defeitos de geração e foram descartados.

### 4.5 Garantias verificadas

O pipeline valida ao final, por asserção automática:

- nenhum texto repetido dentro de uma mesma classe;
- nenhum texto presente nas duas classes;
- treino, validação e teste **disjuntos** (zero interseção) — o dedup global roda antes
  do split, o que é o que impede vazamento e inflação artificial das métricas;
- distribuição de fontes idêntica entre as três partições, dentro de **0,01 ponto
  percentual**.

---

## 5. Etapa 3 — Pipeline de treinamento

Implementado em `project/src/training/`, com núcleo compartilhado e um script fino por
modelo (~30 linhas cada), de modo que os três recebem tratamento idêntico.

```bash
uv run python -m src.training.train_all
```

### 5.1 Modelos

| Modelo | Checkpoint | Parâmetros | Épocas |
|---|---|---|---|
| **BERTimbau Base** | `neuralmind/bert-base-portuguese-cased` | 108,9 M | 4 |
| **Albertina PT-BR** | `PORTULAN/albertina-100m-portuguese-ptbr-encoder` | 139,2 M | 4 |
| **DeBERTinha** | `sagui-nlp/debertinha-ptbr-xsmall` | 40,9 M | 5 |

Os três com **lote efetivo 32**, `max_length` 128, taxa de aprendizado 2e-5, warmup de
10%, dropout 0,1 e a mesma semente — conforme a tabela de hiperparâmetros da
metodologia. A configuração idêntica é proposital: garante que a diferença observada
entre os modelos seja atribuível ao modelo, e não ao regime de treino.

**Hardware:** NVIDIA RTX A2000 12 GB · Intel i7 14ª geração · 32 GB RAM · Windows 11.

### 5.2 Disciplina experimental

O motor de treino segue uma ordem rígida:

```
hardware → semente → dados → modelo → treino
→ avalia VALIDAÇÃO → escolhe o limiar de decisão na VALIDAÇÃO
→ avalia TESTE uma única vez, com o limiar congelado
→ grava artefatos
```

O conjunto de teste é acessado **exatamente uma vez**, ao final, e o limiar de decisão é
escolhido apenas na validação. É o que sustenta a afirmação metodológica de que o teste
permaneceu reservado durante todo o desenvolvimento.

### 5.3 Métricas

Acurácia, acurácia balanceada, precisão, recall e F1 (binário em DD e macro), ROC-AUC,
PR-AUC, MCC e **EER** (*equal error rate*). O critério de aceite da metodologia é
**EER < 30%**.

DD é a classe positiva, de modo que um falso positivo corresponde ao assistente acordar
durante conversa de fundo — o erro crítico do sistema. O EER é obtido por interpolação
da curva ROC, e a implementação foi validada contra o caso analítico de duas
distribuições normais com `d' = 2`, cujo EER teórico é `Φ(−1) = 0,1587` (obtido:
0,1590).

Cada execução grava métricas, matriz de confusão, curvas ROC e DET, predições por
exemplo, **métricas desagregadas por fonte geradora** e um manifesto com configuração,
hardware, versões e SHA do commit.

### 5.4 Situação

✅ **Treinamento concluído** para os três modelos. Resultado no conjunto de teste
(23.747 exemplos, avaliado uma única vez, limiar calibrado apenas na validação):

| modelo | accuracy | f1_dd | f1_macro | roc_auc | eer |
|---|---|---|---|---|---|
| bertimbau | 0,9898 | 0,9898 | 0,9898 | 0,9988 | 0,0109 |
| albertina | 0,9889 | 0,9889 | 0,9889 | 0,9987 | 0,0117 |
| debertinha | 0,9883 | 0,9883 | 0,9883 | 0,9987 | 0,0125 |

Os três modelos ficam muito acima do critério de aceite (EER < 30%) e muito próximos
entre si — a diferença de ~0,001–0,002 entre eles ainda precisa de um teste de
significância (bootstrap pareado, ver Seção 5.5) para saber se é real ou ruído de
amostragem.

### 5.5 Overfitting de calibração — achado a investigar

A leitura do histórico de treino (`trainer_state.json`) de cada modelo mostra o mesmo
padrão nos três: o `eval_loss` atinge o mínimo na época 2–3 e volta a subir depois
(+15% a +24%), enquanto o `train_loss` cai para perto de zero. As métricas discretas de
validação (accuracy, f1_macro) **não** degradam junto — continuam estáveis ou melhoram
ligeiramente, porque o critério de seleção do melhor checkpoint é `f1_macro`, não
`eval_loss`, e por isso o *early stopping* (paciência 2) nunca chegou a disparar.

Isso é overfitting de **confiança/calibração**, não (ainda) de generalização
discreta: o modelo fica cada vez mais confiante nos exemplos de treino sem que isso
piore accuracy/F1 na validação/teste. É um ponto real a documentar e ilustrar
(curvas de loss por época, diagrama de calibração) no capítulo de Resultados, e reforça
a preocupação da limitação abaixo — métricas de ~0,99 num corpus 100% sintético exigem
uma validação externa para confirmar que não é apenas o classificador aprendendo o
estilo dos LLMs geradores.

### 5.6 Baselines clássicos e significância estatística

Implementados em `project/src/baseline/` — TF-IDF (n-gramas 1-3, vocabulário de 50 mil
termos) com `LinearSVC` calibrado e Regressão Logística, ambos com `C` escolhido por
validação cruzada 5-fold no treino, seguindo a mesma disciplina de limiar (calibrado só
na validação, teste avaliado uma única vez):

| modelo | accuracy | f1_dd | f1_macro | roc_auc | eer |
|---|---|---|---|---|---|
| bertimbau | 0,9898 | 0,9898 | 0,9898 | 0,9988 | 0,0109 |
| albertina | 0,9889 | 0,9889 | 0,9889 | 0,9987 | 0,0117 |
| debertinha | 0,9883 | 0,9883 | 0,9883 | 0,9987 | 0,0125 |
| baseline TF-IDF + SVM | 0,9768 | 0,9768 | 0,9768 | 0,9962 | 0,0233 |
| baseline TF-IDF + LogReg | 0,9760 | 0,9760 | 0,9760 | 0,9962 | 0,0239 |

Um bootstrap pareado (`project/src/analysis/bootstrap.py`, 2.000 reamostragens sobre o
teste fixo, substituindo as "5 execuções com sementes diferentes" da metodologia
original — decisão registrada por não haver orçamento de tempo para retreinar)
confirma que a diferença é estatisticamente robusta: **os três Transformers superam os
dois baselines em todas as comparações par-a-par com P(Transformer melhor) = 1,000 em
accuracy/F1/ROC-AUC e 0,000 em EER** (ou seja, em 2.000/2.000 reamostragens o Transformer
venceu). Entre os três Transformers, a diferença já não é tão nítida: BERTimbau supera
DeBERTinha de forma consistente (P = 0,99 em F1, 0,007 em EER), mas BERTimbau vs Albertina
e Albertina vs DeBERTinha não têm diferença estatisticamente confiável (P entre 0,17 e 0,93,
IC cruzando zero) — a hipótese central do TCC1 (Transformer supera baseline linear) fica
confirmada com evidência forte; a escolha entre os três Transformers específicos, não.

Resultado completo em `outputs/comparison_full.md` e `outputs/analysis/bootstrap/pairwise_bootstrap.md`.

---

## 6. Próximos passos

Em ordem de prioridade:

1. ✅ **Baselines clássicos** (F2, semanas 5–7) — concluído, ver Seção 5.6.
2. ✅ **Análise de overfitting e significância estatística** (F4) — concluído, ver
   Seções 5.5 e 5.6. Curvas de loss por época
   (`outputs/analysis/loss_curves/`), gráfico por fonte geradora
   (`outputs/analysis/per_source/`), diagrama de confiabilidade
   (`outputs/analysis/calibration/`) e bootstrap pareado
   (`outputs/analysis/bootstrap/`) já gerados a partir dos artefatos existentes, sem
   retreinar nada.
3. **Pipeline ASR simulado via TTS → Whisper** (F2, semanas 4–6) — próxima prioridade.
   Como não há áudio real gravado para as frases sintéticas, sintetiza-se áudio de uma
   amostra do conjunto de teste via TTS, transcreve-se com `faster-whisper` (já
   instalado) e mede-se a degradação do classificador em função do WER, testando também
   hipóteses n-best (N=1,3,5). **Confirmado viável**: o `ctranslate2` por trás do
   `faster-whisper` expõe n-best real via beam search nativo (parâmetro
   `num_hypotheses` de `Whisper.generate`), só não pela API de alto nível
   `WhisperModel.transcribe()` — não será necessário o fallback por amostragem de
   temperatura.
4. **Validação externa** contra corpora de fala real (CORAA, NURC-SP, TAGARELA,
   disponíveis via Hugging Face Hub) — ver limitação abaixo.
5. **Ablação de normalização** — treinar com `text_raw` (acento e pontuação preservados)
   e comparar. Remover acentos afasta o texto do que os tokenizadores *cased* viram no
   pré-treino e desfaz pares mínimos do português (`está`/`esta`, `é`/`e`); por outro
   lado, aproxima da saída real de um ASR. Os dois números juntos são um resultado, e
   não uma suposição escondida.

### Limitação conhecida

O corpus é **integralmente sintético**. A hipótese do trabalho trata de transcrições de
fala humana, e um classificador pode atingir desempenho alto neste corpus e falhar em
fala real. A mitigação prevista é avaliar os modelos sobre corpora de fala espontânea em
português (CORAA, NURC-SP, TAGARELA) como conjunto de teste externo da classe NDD,
nunca em treino. Essa avaliação ainda não foi executada.

---

## 7. Reprodução

```bash
cd project
uv sync                                          # instala dependências
uv run python -m src.data_prep.run_pipeline      # gera o dataset (~6 s)
uv run python -m src.training.train_all          # treina os 3 modelos
```

O `torch` é resolvido por plataforma: build com CUDA no Windows e Linux, build de CPU no
macOS, sem necessidade de flags. Detalhes de uso, opções de linha de comando e decisões
de implementação estão em [`project/README.md`](project/README.md).

---

## 8. Estrutura do repositório

```
tcc/
├── README.md                    # este documento
├── paper/                       # monografia em LaTeX (abnTeX2)
│   ├── main.tex
│   └── bibliografia.bib
└── project/
    ├── README.md                # documentação técnica detalhada
    ├── pyproject.toml           # dependências (uv)
    ├── data/                    # dataset processado (gerado; não versionado)
    └── src/
        ├── dataset/             # Etapa 1 — geração com LLMs
        │   ├── *_generate.py    #   um script por fonte
        │   ├── prompts/         #   taxonomia de categorias, contextos e tons
        │   └── <fonte>/         #   corpora gerados (dd.jsonl, ndd.jsonl)
        ├── data_prep/           # Etapa 2 — pipeline de preparação (7 estágios)
        └── training/            # Etapa 3 — fine-tuning dos 3 modelos
```

O código está integralmente documentado: **100% dos módulos, classes, funções e métodos**
possuem docstring.
