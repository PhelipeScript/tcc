# Legendas sugeridas das figuras

Uma sugestão por figura, para o `\caption` do LaTeX.
- A primeira frase responde à pergunta da figura. O resto explica como ler e traz as notas que ficaram fora
  da imagem.
- Os números saem dos relatórios citados.
- As figuras estão em `reports/figuras/` (PDF vetorial e PNG) e são geradas por `scripts/figuras.py`.

## Cap. 3 (método)

**M1 · `m1_sistema`.** Caminho de uma fala no sistema em tempo real, do microfone à decisão.
- O detector de voz (Silero VAD) fecha a fala depois de 0,7 s de silêncio.
- O Whisper large-v3-turbo transcreve a fala completa, na GPU.
- Um dos 5 classificadores de texto decide se ela é dirigida ao computador (DD) ou não (NDD): BERTimbau,
  Albertina, DeBERTinha, TF-IDF + Regressão Logística ou TF-IDF + SVM.
- O índice de confiança (lógica fuzzy) sai dos dados do próprio Whisper e é registrado com a decisão, sem
  mudá-la (tracejado).
- Os tempos medidos estão na R7.

**M2 · `m2_funil_corpus`.** Das 289.920 frases das fontes, 116.438 formam o corpus final, com 58.219 por classe.
- A maior perda vem do balanceamento por faixa de tamanho, que iguala DD e NDD em cada faixa, fonte e origem.
- O corpus é dividido em treino (93.150), validação (11.672) e teste (11.616), 80/10/10, sem texto em comum
  entre as partes.
- Fonte: `reports/corpus/principal/report.md`.

**M3 · `m3_treino`.** Nas 9 execuções com correção, a perda de validação chega ao mínimo entre as épocas 2 e 4
e depois volta a subir. Em 8 delas, a F1 macro de validação continua melhorando até a última época.
- Cada linha é uma semente (42, 43 e 44). A linha cheia é a semente usada no sistema, escolhida pela F1 de
  validação; as tracejadas são as outras duas.
- Cada ponto é o fim de uma época. O ponto cheio é o melhor checkpoint, o de maior F1 macro de validação.
- O melhor checkpoint fica na última época em 8 das 9 execuções; no BERTimbau s44, na época 3 de 4.
- A F1 sobe de 0,969-0,977 na época 1 para 0,979-0,983 no melhor checkpoint. Na última época, o ganho é de no
  máximo 0,4 p.p.
- Em 8 das 9 execuções, a menor perda de validação vem antes do melhor checkpoint. A perda também pesa a
  confiança das probabilidades, e a F1 conta só acertos e erros: por isso o checkpoint é escolhido pela F1.
- Os modelos sem correção ficam de fora: a validação deles é a do corpus com o atalho do tamanho (tabela 12).
- Fonte: `reports/treino/treino.json`.

**M4 · `m4_whisper`.** Na GPU, o mlx-whisper transcreve uma fala em 0,39-0,66 s, de 1 a 20 s de fala. Na CPU,
o faster-whisper leva de 2,8 a 3,5 s em fp32 e de 4,9 a 6,7 s em int8.
- Mediana de 5 repetições por duração (2 nas falas de 20 s), com o mesmo modelo (Whisper large-v3-turbo) e a
  mesma decodificação gulosa.
- No macOS, o faster-whisper roda só na CPU, aqui com 4 threads (os 4 núcleos de desempenho do M5). Neste Mac,
  o int8 foi mais lento que o fp32.
- O tempo quase não muda com a duração da fala, porque o Whisper sempre processa uma janela de 30 s.
- Acima da diagonal, transcrever leva mais tempo que a própria fala: o int8 fica acima dela nas falas de até
  ~5 s, e o fp32, nas de até ~3 s.
- Em 5 s de fala: 0,45 s no mlx (RTF 0,090), 2,93 s no fp32 (0,586) e 5,23 s no int8 (1,045) (tabela 9).
- Mede só o tempo, não a qualidade da transcrição.
- Medido na tomada, com carga de 1,4 no início, com fala sintetizada pelo macOS (`say`).
  Na bateria (`reports/benchmark/benchmark-20261104-0737.json`), os tempos em 5 s foram parecidos: 0,43, 2,62
  e 5,30 s.
- Fonte: `reports/benchmark/benchmark-20261105-1623.json`.

## Resultados

**R1 · `r1_vies_comprimento`.** No corpus inicial, o tamanho da frase revelava a classe.
- Frases de 3 palavras eram 99,9% dirigidas ao computador, e as de 18 a 24 palavras, só 1,0%.
- Só o tamanho previa a classe com AUC de 0,86.
- No corpus final, as duas classes têm o mesmo número de frases em cada faixa, e a AUC cai para 0,515.
- Não há frases de 25-50 palavras no corpus corrigido.
- Fonte: `reports/corpus/{controle,principal}/manifest.json`.

**R2 · `r2_teste_interno`.** No teste interno (11.616 frases), os três Transformers erram menos que os TF-IDF.
- Os modelos treinados sem a correção erram cerca de quatro vezes mais nesse teste (de 7,8% a 8,4%), que não
  tem o atalho do tamanho. No teste do próprio corpus, eles erram de 1,1% a 1,5%: parte do acerto vinha do
  atalho (tabela 2).
- Nos Transformers, o ponto é a média de 3 sementes, e o rótulo traz o desvio padrão em p.p. (de 0,0 a 0,1).
  O desvio é menor que o marcador, por isso não há barra. Na DeBERTinha, a acurácia é a mesma nas 3 sementes
  (267 frases erradas em cada; tabela 2). No EER dela, o "± 0,0" é arredondamento: o desvio é de 0,04 p.p.
- A diferença entre Transformers e TF-IDF é significativa no bootstrap pareado (tabela 4).
- Fonte: `reports/experimento/teste_interno.json`.

**R3 · `r3_correcao`.** A correção do viés de comprimento reduz as falsas ativações fora do laboratório nos três
conjuntos, de 22-32% para 5-8%.
- Os três conjuntos são as transcrições dos corpora públicos (2.912 frases), a voz do autor (104 falas não
  dirigidas) e o áudio dos corpora públicos (600 trechos).
- Cada linha liga o mesmo Transformer sem correção (quadrado) e com correção (círculo).
  - Transcrições: de 22,0-30,8% para 5,4-6,3%.
  - Voz do autor: de 25,0-29,8% para 5,8-7,7%.
  - Áudio dos corpora públicos: de 23,7-32,3% para 5,2-6,5%.
- A queda é significativa nos 9 pares (bootstrap pareado). Na voz do autor, os intervalos são largos porque há
  só 44 frases não dirigidas distintas.
- Na voz do autor, 80 das 104 falas são frases do teste lidas em voz alta. As 24 do roteiro de conversa, mais
  perto do uso real, deram 0 ativações nos 3 Transformers com correção, contra 5, 5 e 7 sem correção.
- Nas transcrições, o modelo sem correção é o da semente 42, e o com correção é o da semente escolhida pela
  validação. Nos dois áudios, os modelos sem correção julgaram as mesmas transcrições do sistema ao vivo.
- Os limiares dos modelos sem correção, escolhidos na validação do corpus deles, vão de 0,03 a 0,49. Com 0,5
  em todos, a diferença continua nos dois áudios da figura: 23,7-29,8% sem correção contra 5,2-8,7% com
  correção (tabela 13). As transcrições não foram reavaliadas com 0,5.
- A correção inclui as frases novas (com os negativos difíceis) e o balanceamento por faixa de tamanho. O
  experimento compara os dois corpora inteiros e não separa o efeito de cada parte.
- Fontes: `reports/externa_texto/resultado.json` e `reports/pipeline_{gravacoes,publico}/comparacao.json`.

**R4 · `r4_falsas_ativacoes_por_tamanho`.** Com a correção, as falsas ativações caem sobretudo nas frases curtas:
nas de 2 palavras, de 90% para 9%.
- Nas frases longas acontece o contrário: a partir de 11 palavras, com a correção há mais falsas ativações que
  sem ela, e a diferença cresce com o tamanho, de 4,7% contra 1,9% (14-17 palavras) e 7,3% contra 1,9%
  (18-24) até 10,6% contra 0,5% (25-50).
- As faixas até 24 palavras estão no treino com correção. A de 25-50 não aparece no corpus com correção; o
  corpus sem correção tem 281 frases não dirigidas nela.
- No corpus sem correção, frase longa era quase sempre NDD (89% das de 14-17 palavras e 99% das de 18-24).
  Com a correção, o tamanho deixa de indicar a classe.
- A queda vem da correção como um todo: as frases novas, entre elas negativos difíceis curtos, e o
  balanceamento por faixa. O experimento não separa as duas partes.
- Os dados são as 2.900 transcrições dos corpora públicos (CORAA, NURC-SP e TAGARELA), todas não dirigidas.
  As 12 frases com mais de 50 palavras foram omitidas.
- Cada curva é a média dos três Transformers.
- Fonte: `reports/externa_texto/resultado.json`.

**R5 · `r5_asr`.** O Whisper erra mais na fala real do que na voz sintética. O TF-IDF já perde acerto com erros
moderados de transcrição; na média, os Transformers só perdem acima de 0,3 de WER.
- (a) Taxa de erro de palavras (WER) por condição.
  - Voz sintética: 4,9%, nas 740 frases.
  - Voz do autor, nas frases do teste: 13,7% (sala quieta) a 22,0% (ruído de fundo).
  - Roteiro de conversa: 3,1%. Nas frases curtas do teste, um erro já pesa muito no WER.
  - Áudio dos corpora públicos: 27,4%, em 600 trechos de conversa espontânea de outros falantes. Sem as pausas
    preenchidas ("eh", "ahn"), 27,1%. Repetições ("na na") e números em algarismos também contam como erro.
- (b) Queda do acerto do classificador, do texto lido para a transcrição, nos mesmos itens, por faixa de WER.
  - Os dados juntam a voz sintética e a voz do autor (924 itens).
  - Acima de 0,3 de WER (69 itens), o acerto cai 12,3 p.p. no TF-IDF e 6,8 p.p. nos Transformers. As quedas
    não têm intervalo de confiança.
  - Entre 0,1 e 0,3 (126 itens), o TF-IDF já perde acerto, e a média dos Transformers fica perto de zero.
    Por voz, nessa faixa: o TF-IDF perde 9,0-10,3 p.p. na sintética e 4,2 p.p. na do autor, o mesmo que o
    BERTimbau (`reports/asr_*/hipotese.md`).
  - O áudio público fica fora de (b), porque só tem falas não dirigidas. Nele, da transcrição do corpus para
    a do Whisper, as falsas ativações sobem de 0,5 p.p. (BERTimbau) a 1,3 p.p. (TF-IDF), sem diferença
    significativa entre os modelos (tabela 7).
- Fontes: `reports/asr_tts/`, `reports/asr_gravacoes/` e `reports/asr_publico/`.

**R6 · `r6_sistema_completo`.** No sistema completo em tempo real, nenhum modelo supera o BERTimbau de forma
significativa. DeBERTinha e Albertina não diferem dele de forma significativa, e os TF-IDF ficam atrás em alguns
casos.
- Cada ponto é a diferença para o BERTimbau, em pontos percentuais, com IC 95% do bootstrap pareado.
  - O cabeçalho indica o lado em que o modelo é pior que o BERTimbau.
  - Os painéis da mesma medida têm a mesma escala.
- As diferenças significativas são todas dos TF-IDF:
  - voz do autor, pedidos atendidos: -7,5 p.p. [-16,3; -1,3], nos dois. Eles perdem os mesmos 10 pedidos, e
    o limite do IC fica a 1 leitura do zero;
  - voz sintética, pedidos atendidos: -3,2 p.p. [-5,9; -0,5], na Regressão Logística;
  - voz sintética, falsas ativações: +3,2 p.p. [+0,8; +5,9], no SVM.
- Intervalos que terminam exatamente no zero não contam como diferença.
- Todos os modelos receberam o mesmo áudio e a mesma transcrição (semente fixa do Whisper).
- Na voz do autor, o bootstrap sorteia as frases, porque as mesmas frases foram lidas nos 4 cenários. Por isso
  os intervalos são mais largos. Nos outros conjuntos, ele sorteia os itens.
- Fontes: `reports/pipeline_{gravacoes,tts,publico}/comparacao.json`.

**R7 · `r7_latencia`.** Do fim da fala até a decisão, a maior parte do tempo é a espera de silêncio do detector de
voz (0,7 s). Depois vem o Whisper (0,42-0,43 s), e o classificador leva menos de 50 ms.
- As barras mostram a mediana de cada parte na execução com o BERTimbau. O texto ao lado traz a mediana e o
  percentil 95 do tempo total.
- O resto (filas e processamento do detector de voz) tem percentil 95 de 14 a 16 ms e não aparece nas barras.
- O percentil 95 do áudio público (1,91 s) vem das 39 falas em que o Whisper fez uma nova tentativa. Sem elas,
  ele fica em 1,25 s.
- Os números estão na tabela 9.
- Com os outros 4 modelos, a mediana do total fica entre 1,12 e 1,18 s.
- Fontes: `reports/pipeline_{gravacoes,tts,publico}/bertimbau/registros.jsonl`.
