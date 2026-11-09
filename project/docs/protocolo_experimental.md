# Protocolo experimental

Este texto descreve o procedimento. Os números estão nos relatórios em `reports/`, gerados pelos comandos
citados aqui.

## 1. Corpus (100% sintético)

**Fontes**
- **Corpus inicial** (origem `legacy`), copiado com manifesto (SHA-256 de cada arquivo).
- **Frases novas da correção** (origem `mitigation`), registradas linha a linha: provedor, modelo, versão
  do prompt, categoria, contexto, tom e data.
- Os corpora públicos **não** entram no corpus; são usados só na avaliação.

**Viés de comprimento no corpus inicial**
- Só o tamanho da frase prevê a classe com AUC de 0,86, no corpus sem correção (`reports/corpus/controle/`).
- A mediana é de 7 palavras no DD e de 11 no NDD.

**Correção do viés: geração em pares** (estratégia `comprimento`, prompt `comprimento-v2`)
- Para cada fonte e faixa, o mesmo modelo e a mesma versão de prompt geram **as duas classes, em
  quantidades iguais**.
  - As faixas são as curtas (2, 3, 4, 5 e 6 palavras) e as longas (11-13, 14-17 e 18-24).
  - Há um prompt por classe (`ddsd/corpus/generation/prompts.py`), e cada lote gera uma classe só.
- Cada lote passa por:
  - **filtros determinísticos**: palavra de ativação, casa inteligente, faixa, duplicata e conflito com a
    outra classe;
  - **autoverificação do rótulo pelo próprio gerador**: o mesmo modelo recebe o lote (até 10 frases) numa
    chamada nova, de um turno, em temperatura 0, sem ver o rótulo pretendido, e o reclassifica.
    - Só ficam as frases que ele classifica com o rótulo pretendido.
    - Se a resposta vier inválida, o lote inteiro é descartado.
    - O prompt dessa estratégia manda responder DD na dúvida.
- **Modelos geradores:** gemma-4-e4b (LM Studio local, API nativa sem raciocínio), gpt-oss-120b e
  qwen3.8-27b (Groq), Gemini 3.5 Flash e Claude (subagentes). As cotas gratuitas das APIs limitaram o volume.

**Negativos difíceis** (estratégia `negativos_dificeis`, prompt `negativos_dificeis-v1`)
- **Motivo:** na geração por comprimento, a autoverificação descartou 17,3% das NDD julgadas, contra 0,7%
  das DD (`reports/corpus/autoverificacao.md`).
  - Nessa estratégia, o prompt da autoverificação manda responder DD na dúvida, então um NDD curto com forma
    de pedido tende a ser descartado.
- O par tem a mesma forma (imperativo curto, 2-6 palavras), e a diferença está **na ação**:
  - NDD: pedido a uma pessoa, com ação física ou pessoal ("passa o copo aí");
  - DD: ação de software no computador.
- A autoverificação desta estratégia classifica pela ação, e não pela forma.
- Cada execução grava a sua própria fonte (`<provedor>_negdif`). Assim, o balanceamento
  `faixa_fonte_origem` casa as duas classes dentro da mesma fonte.

**Frases dos subagentes Claude** (`ddsd generate ingest`)
- Passam pelos mesmos filtros determinísticos (palavra de ativação, casa inteligente, faixa, duplicata,
  conflito).
- **Não passam pela autoverificação.** O rótulo pretendido é aceito como veio.
- A revisão humana mede a qualidade desses rótulos.

**Alcance da autoverificação** (`ddsd corpus autoverificacao`, `reports/corpus/autoverificacao.md`)
- Passaram por ela 17.904 das 116.438 frases do corpus final (15%).
- O corpus inicial (82%) e as 3.144 frases do Claude não passaram.
- **Taxa de discordância** nos lotes com log, que são 62% das frases verificadas:
  - no geral, 1,0% dos DD e 13,0% dos NDD;
  - na estratégia de comprimento, 0,7% e 17,3%;
  - nos negativos difíceis, 1,5% e 3,6%.
- **Termo:** "autoverificação do rótulo pelo próprio gerador". Lembra o *LLM-as-a-judge*, mas só como
  analogia: o avaliador é o mesmo modelo que gerou a frase, não um juiz independente.
- O critério de anotação DD/NDD, com as exclusões, está em `docs/criterio_anotacao.md`.

**Construção** (`ddsd corpus build`, determinística; relatório em `data/corpus/build/report.md`)
1. Normalização única (`ddsd/text/normalization.py`).
2. Filtros: mais de 5 caracteres, de 2 a 50 palavras, só ASCII depois da normalização, e sem lista
   serializada (defeito de gerador encontrado no corpus inicial, com 70 registros).
3. Dedup exato por classe e remoção de conflitos (texto nas duas classes sai das duas).
4. Dedup aproximado **exato** (sem perder pares), por cosseno TF-IDF > 0,95 com filtragem de prefixo
   (All-Pairs). Pares entre classes diferentes saem como conflito.
5. Balanceamento `faixa_fonte_origem`: 1:1 dentro de cada célula (faixa de comprimento, fonte, origem).
   Por construção, nem o comprimento (dentro da faixa), nem o modelo gerador, nem o estilo do prompt
   predizem a classe.
6. Split 80/10/10 estratificado por classe, fonte e faixa (semente 8), sem texto em comum entre as
   partições.

**Diagnósticos no relatório**: AUC de comprimento, de fonte e de origem por faixa, antes e depois,
para o conjunto e para cada partição.

**Congelamento** (`ddsd corpus freeze`)
- Confere o SHA-256 de cada fonte contra o manifesto do build. Recusa o congelamento se a geração
  anexou frases depois do build.
- Arquiva as fontes exatas em `sources.tar.gz`, junto das partições.
- Copia o relatório e o manifesto para `reports/corpus/{principal,controle}/`.
- O congelamento foi feito em 2026-11-01, com 116.438 frases em cada corpus: o com correção (`principal`)
  e o sem correção (`controle`). Os números completos estão nesses relatórios.

**Corpus sem correção** (`config/corpus_controle.toml`; no código e nas pastas, "controle")
- Só o corpus inicial, com balanceamento **global** 1:1.
- **Mesmo tamanho** do corpus com correção, para que só a composição mude.
- **Sem os textos do teste do corpus com correção**, nem as quase-duplicatas deles (9.527 textos). Assim
  os modelos com e sem correção podem ser comparados no mesmo teste.

**Revisão humana**
- `ddsd corpus annotate`: ~300 frases estratificadas, cegas, em ordem aleatória.
- `ddsd corpus agreement`: kappa de Cohen geral, por origem, por fonte e por faixa, mais as divergências.

## 2. Treino (Mac M5, MPS)

**Hiperparâmetros**: os mesmos da metodologia.
- lr 2e-5, lote 32, épocas 4 (BERTimbau, Albertina) e 5 (DeBERTinha);
- `max_length` 128, warmup de 10%, dropout 0,1 no atributo correto de cada família;
- weight decay 0,01, early stopping com paciência 2 sobre a F1 macro.

**Precisão recomendada por modelo, sem contornos**
- BERTimbau e DeBERTinha: bf16 misto.
- Albertina: fp32. O bf16 falha no MPS por um overflow do `masked_fill` do DeBERTa V1.
- O motivo fica registrado no manifesto.

**Disciplina**: o limiar é escolhido só na validação, e o teste é avaliado uma única vez.

**Rastreabilidade**: os manifestos de treino, de baseline, de benchmark e de build do corpus registram a
fonte de energia, a carga da bateria e o Modo de Pouca Energia (os da DeBERTinha s42 e dos corpora
congelados não têm esse registro).

**Avaliação final** com o **melhor checkpoint lido do disco, em fp32 sem autocast**, exatamente o
artefato que o pipeline carrega.
- Motivo: com precisão mista, o Accelerate envolve o `forward` em autocast, e avaliar o modelo em memória
  daria probabilidades até ~0,01 diferentes das da inferência fp32.
- Paridade verificada (`ddsd eval parity`): decisões idênticas em ≥ 99,9% dos casos.

**Execuções**
- 3 sementes (42, 43, 44) por modelo com correção, mais 1 semente de cada modelo sem correção
  (`scripts/treino_completo.sh`, retomável).
- **Escolha da semente** (`ddsd.evaluation.runs.best_seed`, critério fixado no código antes do treino):
  - critério: a maior F1 macro de **validação** com limiar 0,5 (`metrics_val.json`), o mesmo critério
    usado para escolher o checkpoint;
  - a execução escolhida é a publicada no Hugging Face, que o pipeline usa (`config/pipeline.toml`), e
    entra no bootstrap pareado e nas avaliações externas;
  - o limiar de decisão de cada execução continua o escolhido na validação;
  - o teste interno é reportado com todas as sementes (média ± desvio).

## 3. Avaliação

| Experimento | Comando | Saída |
|---|---|---|
| Teste interno: média ± desvio entre sementes, por faixa, e os modelos sem correção no mesmo teste | `ddsd eval internal` | `reports/experimento/teste_interno.*` |
| Matriz de confusão e erros mais confiantes no teste interno | `ddsd eval erros` | `reports/experimento/erros.*` |
| Baselines TF-IDF (1 a 3-gramas, 50 mil termos) + SVM linear calibrado / Regressão Logística | `ddsd eval baselines` | `models/baseline_*` |
| Bootstrap pareado no teste interno (2.000 reamostras, alinhamento verificado) | `ddsd eval bootstrap` | `reports/experimento/bootstrap.*` |
| Transcrições dos corpora públicos (CORAA, NURC-SP e TAGARELA), sem sobreposição com o corpus | `ddsd eval external-prepare` / `external` | `reports/externa_texto/` |
| Impacto do ASR: voz sintética (3 vozes pt-BR), voz do autor e áudio dos corpora públicos | `ddsd eval asr tts\|gravacoes\|publico` | `reports/asr_*/resumo.*` |
| A hipótese nas transcrições do Whisper e a queda do acerto por faixa de WER | `ddsd eval asr-hipotese tts\|gravacoes` | `reports/asr_*/hipotese.*` |
| Robustez ao ASR: queda do acerto (texto → transcrição) comparada entre modelos | `ddsd eval asr-robustez tts\|gravacoes\|publico` | `reports/asr_*/robustez.*` |
| Sistema completo em tempo real, um modelo por execução | `ddsd eval pipeline gravacoes\|tts\|publico --classificador X` (todos os 5: `scripts/sessao_pipeline.sh`) | `reports/pipeline_*/<modelo>/` |
| Comparação no sistema completo: as 5 execuções ao vivo e os 3 modelos sem correção | `ddsd eval comparar gravacoes\|tts\|publico` | `reports/pipeline_*/comparacao.*` |
| Sessão ao vivo com roteiro, pontuada contra o roteiro | `ddsd run --gravar`, depois `ddsd eval sessao data/sessoes/<data-hora>` | `reports/sessao_ao_vivo/` |
| Custo: parâmetros, tamanho, treino, memória e inferência | `ddsd eval custo` | `reports/custo/` |
| Desempenho no Mac (Whisper, classificadores, frontend) | `ddsd benchmark` | `reports/benchmark/` |
| Autoverificação do rótulo na geração | `ddsd corpus autoverificacao` | `reports/corpus/autoverificacao.*` |

**Notas de método**
- O WER usa `normalize_text` nos dois lados.
- **Sessão ao vivo com roteiro** (`live_session.py`):
  - as falas detectadas são casadas com as linhas lidas do roteiro, em ordem e pelo texto (similaridade de
    caracteres acima de 0,5), aceitando junção e divisão de até 3 e falas fora do roteiro;
  - a unidade é o grupo casado: DD se alguma linha for DD, ativado se alguma fala foi decidida DD;
  - as falas fora do roteiro (improvisos) não entram nas taxas;
  - cada erro recebe a origem: segmentação, classificador (erra também no texto do roteiro) ou transcrição
    (acerta no roteiro; junta o erro do ASR e o desvio da leitura, que só o áudio separa);
  - julgar de novo as transcrições precisa repetir todas as decisões da sessão.
- **Corpora públicos:** todo enunciado é NDD por suposição de domínio, e a métrica é a taxa de falsas
  ativações.
- **Ignorar transcrições pouco confiáveis** (confiança baixa → NDD): a regra é só avaliada (coluna
  "política" dos relatórios). Ela não está ligada no sistema.
- **Impacto do ASR:** o texto lido e a transcrição do Whisper são comparados nos **mesmos** itens
  (pareado).
  - A queda do acerto de cada item é: acerto no texto - acerto na transcrição.
  - Ela é resumida por faixa de WER e correlacionada com o WER (Spearman).
  - **Áudio dos corpora públicos:** os 600 trechos do sistema completo, transcritos inteiros (sem o detector
    de voz), como nos outros conjuntos. A referência é a transcrição do próprio corpus. Todos são NDD, então a
    medida é a taxa de falsas ativações na referência e na transcrição do Whisper.
  - **WER secundário:** sem as pausas preenchidas (`FILLED_PAUSES` em `asr_impact.py`: "eh", "ahn"…),
    removidas dos dois lados. "ha" fica fora da lista porque, normalizado, colide com o verbo "há".
- **Voz do autor:** 40 frases do teste em 4 cenários (sala quieta, ruído de fundo, iPhone e ~2 m), mais o
  roteiro de conversa (`config/roteiro_conversa.txt`).
  - O roteiro tem 24 falas não dirigidas, lidas como numa conversa com uma pessoa na sala. Nenhuma é igual
    ou quase igual a uma frase do corpus.
  - A fala livre (sem roteiro) fica à parte:
    - no impacto do ASR, entra só a taxa de falsas ativações;
    - no sistema completo, entram as falsas ativações por fala detectada e por minuto, porque cada
      arquivo tem várias falas.
- **Sistema completo, um modelo por execução:**
  - os 5 modelos com correção (BERTimbau, os 2 TF-IDF, DeBERTinha e Albertina) rodam ao vivo, cada um no
    seu processo e na mesma ordem;
  - o áudio toca no ritmo normal, no mesmo buffer do microfone;
  - os 3 modelos sem correção julgam as transcrições da execução do BERTimbau, pela regra do próprio
    decisor. Julgar de novo com o BERTimbau tem de reproduzir 100% das decisões (paridade).
- **Semente fixa do Whisper** (`[asr] seed`):
  - Quando a 1ª transcrição falha nos testes de qualidade (compressão ou log-prob), o Whisper transcreve
    de novo com sorteio (temperatura > 0).
  - O mlx-whisper compila esse sorteio (`mx.compile`), que guarda a chave aleatória da primeira chamada.
    Com a semente fixa na abertura do Whisper, o sorteio fica igual entre execuções.
  - **Verificação:** execuções repetidas do mesmo conjunto, uma por processo, deram transcrições
    idênticas, conferidas item a item (`ddsd eval comparar`).
  - A chave depende da ordem em que as novas tentativas aparecem. Por isso uma repetição sempre refaz o
    conjunto inteiro, na mesma ordem.
- **Comparações entre os modelos**, todas pareadas nos mesmos itens, com bootstrap de 2.000 reamostras:
  - **Unidade sorteada:** na voz do autor, as mesmas frases foram lidas nos 4 cenários, e os erros de um
    modelo se repetem na mesma frase. Por isso o bootstrap sorteia as frases, e as leituras de cada frase
    entram juntas (`ddsd.evaluation.resampling`). Nos outros conjuntos, cada frase aparece uma vez e o
    sorteio é por item.
  - **Teste interno e transcrições do Whisper:** acurácia, F1 macro e EER.
  - **Transcrições dos corpora públicos:** falsas ativações. Os Transformers são comparados entre si, com o
    TF-IDF e com a própria versão sem correção.
  - **Variação entre sementes:** média e desvio, porque a semente escolhida pela validação não representa
    sozinha o modelo.
  - **Sistema completo:** cada modelo contra o BERTimbau e cada Transformer contra a própria versão sem
    correção.
  - **Latência:** a diferença entre os modelos é só a do classificador. As falas são as mesmas, então o
    tempo do Whisper fala a fala também mede a deriva entre as execuções.

## 4. Limitações conhecidas

- **Corpus 100% sintético:** ainda há diferença de estilo entre o texto de LLM e a fala transcrita.
- **Rótulos do corpus inicial:** 82% do corpus final vem do corpus inicial.
  - Esses rótulos não passaram pela autoverificação.
  - Os falsos positivos mais confiantes do BERTimbau no teste interno vêm dele e parecem rótulos errados,
    como "muda a fonte do titulo" rotulado NDD (`reports/experimento/erros.md`).
  - O teste interno mede a concordância com esses rótulos.
- **Diversidade dos geradores:** as cotas gratuitas das APIs limitaram os geradores. Das 21.048 frases
  novas do corpus final, as fontes são:

  | Gerador | Participação |
  |---|---|
  | gemma-4-e4b (LM Studio local) | 51% |
  | qwen3.8-27b | 18% |
  | gpt-oss-120b | 15% |
  | Claude | 15% |
  | Gemini 3.5 Flash | 0,3% |
- **Autoverificação parcial:** o avaliador é o próprio gerador, não um juiz independente. Ela cobre só 15%
  do corpus final.
- **Frases truncadas:** são as que terminam em artigo ou preposição, como "passa a".
  - No corpus congelado, são 1,8% do NDD e 0,8% do DD de 2-6 palavras, e 0,7% e 0,3% no corpus inteiro.
  - O efeito é desprezível, então elas não foram filtradas.
- **Teste interno perto do teto:** a F1 macro fica entre 0,977 e 0,982, e os Transformers diferem por
  0,3-0,6 p.p. Quem separa os modelos são as conversas reais e a fala.
- **A comparação com o corpus sem correção mede a correção inteira.**
  - Os dois corpora diferem em duas coisas ao mesmo tempo: as frases novas e o balanceamento por faixa (em
    vez do global).
  - O efeito de cada parte não é separado.
- **Frases longas:** nas transcrições públicas, a partir de 11 palavras, a correção tem mais falsas ativações
  que o corpus sem ela, e a diferença cresce com o tamanho.
  - Na média dos 3 Transformers (fig. R4): 4,7% contra 1,9% nas frases de 14-17 palavras e 7,3% contra
    1,9% nas de 18-24. Essas faixas estão no treino com correção, balanceadas 1:1.
  - Nas de 25-50 palavras, 9-12% com a correção. Esse tamanho não aparece no corpus com correção. Nessa
    faixa, o TF-IDF fica em 2,4%, e os modelos sem correção, em até 1%.
  - No corpus sem correção, frase longa era quase sempre NDD: 89% das de 14-17 palavras e 99% das de 18-24.
    A subida é compatível com a retirada desse atalho, e não só com a falta de frases de 25-50 palavras.
- **Sementes e limiar:**
  - No BERTimbau e na DeBERTinha, a semente escolhida pela validação (s44) é a de mais falsas ativações nas
    transcrições públicas.
  - O limiar escolhido na validação sintética (por exemplo, 0,395 na DeBERTinha s44) se transfere de forma
    imperfeita para a fala real.
- **BERTimbau como principal:** é o melhor no teste interno.
  - Nas transcrições públicas, porém, a semente escolhida tem 0,96 p.p. a mais de falsas ativações que a da
    Albertina [0,14; 1,82].
  - Na média das 3 sementes, a diferença cai para ≈ 0,2 p.p. O desvio entre sementes (0,65-0,85 p.p.) tem a
    mesma ordem das diferenças entre os modelos, então a significância depende das sementes escolhidas.
- **Melhor checkpoint:** no BERTimbau s44, foi o da época 3 de 4 (passo 8.733). Nas outras 11 execuções,
  foi o da última época.
- **Energia e temperatura:**
  - Algumas execuções de treino rodaram, total ou parcialmente, na bateria, todas sem Modo de Pouca
    Energia. Os manifestos registram quais.
  - No Mac sem ventoinha, o *throttling* térmico reduziu a velocidade do treino nas execuções longas. A
    queda aparece nos logs de treino (passos/s), que ficam fora do git; não há medida versionada. As métricas
    não dependem disso.
- **Nova tentativa do Whisper:** a transcrição sorteada é reprodutível com a semente, mas não é melhor.
  - Essas falas tendem a sair degradadas, com confiança baixa.
  - Também são elas que formam a cauda da latência.
- **Revisão humana:** um único anotador, o autor.
  - A anotação foi cega para rótulo e fonte, mas não para o objetivo do trabalho.
  - O kappa (0,873) mede a concordância com os rótulos dos LLMs, não entre humanos.
- **Voz do autor:** um único falante, com as mesmas 40 frases nos 4 cenários (cada erro vale 2,5 p.p.) e 24
  falas no roteiro.
  - Isso basta para mostrar ordens de grandeza, mas não para comparar cenários com significância.
  - As comparações sorteiam as frases (64), não as 184 leituras. Com isso, a hipótese não é significativa em
    acurácia, F1 e EER na voz do autor.
- **Falas longas:** acima de 30 s, são cortadas em continuações.
- **Vários falantes:** sem diarização, viram uma única fala.
- **Determinismo:** o MPS não é determinístico bit a bit, o que afeta o treino.
- **Longa duração:** não houve teste de 30 min sem interrupção. As medidas disponíveis são as latências das
  execuções em tempo real (`reports/pipeline_*`) e a sessão com roteiro no sistema final, de 9,5 min
  (`reports/sessao_ao_vivo/`).
