# Tabelas do TCC

Geradas por `scripts/tabelas_tcc.py` a partir de `reports/`.

## Tabela 1. Revisão humana cega (300 frases, um anotador)

| Recorte | Frases | Concordância | Kappa de Cohen |
|---|---|---|---|
| Geral | 300 | 93,7% | 0,873 |
| Corpus inicial | 182 | 94,5% | 0,890 |
| Frases novas da correção | 118 | 92,4% | 0,848 |
| Geração por tamanho | 73 | 94,5% | 0,890 |
| Negativos difíceis | 45 | 88,9% | 0,779 |

| Rótulo do LLM | Autor: DD | Autor: NDD |
|---|---|---|
| DD | 137 | 13 |
| NDD | 6 | 144 |

Fonte: `reports/corpus/anotacao_phelipe.md`.

## Tabela 2. Teste interno (11.616 frases)

| Modelo | Acurácia | F1 macro | EER |
|---|---|---|---|
| BERTimbau | 0,9822 ± 0,0010 | 0,9822 ± 0,0010 | 0,0178 ± 0,0006 |
| Albertina | 0,9788 ± 0,0012 | 0,9788 ± 0,0012 | 0,0219 ± 0,0009 |
| DeBERTinha | 0,9770 ± 0,0000 | 0,9770 ± 0,0000 | 0,0226 ± 0,0004 |
| TF-IDF + Regressão Logística | 0,9609 | 0,9609 | 0,0393 |
| TF-IDF + SVM | 0,9615 | 0,9615 | 0,0401 |
| BERTimbau sem correção | 0,9221 | 0,9218 | 0,0832 |
| Albertina sem correção | 0,9201 | 0,9199 | 0,0878 |
| DeBERTinha sem correção | 0,9155 | 0,9152 | 0,0882 |

Transformers com correção: média ± desvio de 3 sementes. TF-IDF: 1 execução. Sem correção: 1 semente, avaliada no mesmo teste. Precisão e sensibilidade estão na tabela 3. DeBERTinha: a acurácia é a mesma nas 3 sementes (267 frases erradas em cada); o desvio da F1 macro fica abaixo de 0,00005. Fonte: `reports/experimento/teste_interno.md`.

**Modelos sem correção nos dois testes**

| Modelo sem correção | Acurácia no teste do próprio corpus | Acurácia no teste com correção | EER no próprio teste | EER no teste com correção |
|---|---|---|---|---|
| BERTimbau sem correção | 0,9887 | 0,9221 | 0,0114 | 0,0832 |
| Albertina sem correção | 0,9865 | 0,9201 | 0,0138 | 0,0878 |
| DeBERTinha sem correção | 0,9855 | 0,9155 | 0,0138 | 0,0882 |

O teste do próprio corpus sem correção ainda tem o atalho do tamanho; o teste com correção, não. A queda mostra quanto do acerto vinha do atalho.

## Tabela 3. Matriz de confusão no teste interno

| Modelo | DD aceito | DD perdido | NDD aceito (falsa ativação) | NDD ignorado | Precisão (DD) | Sensibilidade (DD) |
|---|---|---|---|---|---|---|
| BERTimbau | 5.719 | 89 | 106 | 5.702 | 98,18% | 98,47% |
| Albertina | 5.706 | 102 | 130 | 5.678 | 97,77% | 98,24% |
| DeBERTinha | 5.705 | 103 | 164 | 5.644 | 97,21% | 98,23% |
| TF-IDF + Regressão Logística | 5.586 | 222 | 232 | 5.576 | 96,01% | 96,18% |
| TF-IDF + SVM | 5.616 | 192 | 255 | 5.553 | 95,66% | 96,69% |

Semente escolhida pela validação. Fonte: `reports/experimento/erros.md`.

## Tabela 4. Transformers × TF-IDF: bootstrap pareado

**Teste interno**

| Comparação | Acurácia (p.p.) | F1 macro (p.p.) | EER (p.p.) | F1 significativo? |
|---|---|---|---|---|
| BERTimbau - TF-IDF + Regressão Logística | +2,2 [+1,9; +2,6] | +2,2 [+1,9; +2,6] | -2,2 [-2,6; -1,9] | sim |
| BERTimbau - TF-IDF + SVM | +2,2 [+1,9; +2,5] | +2,2 [+1,9; +2,5] | -2,3 [-2,6; -1,9] | sim |
| Albertina - TF-IDF + Regressão Logística | +1,9 [+1,6; +2,3] | +1,9 [+1,6; +2,3] | -1,9 [-2,2; -1,5] | sim |
| Albertina - TF-IDF + SVM | +1,8 [+1,5; +2,2] | +1,8 [+1,5; +2,2] | -1,9 [-2,3; -1,6] | sim |
| DeBERTinha - TF-IDF + Regressão Logística | +1,6 [+1,3; +2,0] | +1,6 [+1,3; +2,0] | -1,7 [-2,1; -1,4] | sim |
| DeBERTinha - TF-IDF + SVM | +1,5 [+1,2; +1,9] | +1,5 [+1,2; +1,9] | -1,8 [-2,1; -1,4] | sim |

**Transcrições da voz sintética** (740 itens)

| Comparação | Acurácia (p.p.) | F1 macro (p.p.) | EER (p.p.) | F1 significativo? |
|---|---|---|---|---|
| BERTimbau - TF-IDF + Regressão Logística | +2,6 [+0,8; +4,3] | +2,6 [+0,8; +4,3] | -2,4 [-4,1; -0,8] | sim |
| BERTimbau - TF-IDF + SVM | +2,5 [+0,8; +4,2] | +2,5 [+0,8; +4,2] | -2,7 [-4,4; -1,0] | sim |
| Albertina - TF-IDF + Regressão Logística | +3,4 [+1,8; +5,1] | +3,4 [+1,8; +5,2] | -2,8 [-4,5; -1,1] | sim |
| Albertina - TF-IDF + SVM | +3,4 [+1,8; +5,1] | +3,4 [+1,8; +5,1] | -3,1 [-4,8; -1,4] | sim |
| DeBERTinha - TF-IDF + Regressão Logística | +2,6 [+1,1; +4,3] | +2,6 [+1,1; +4,3] | -2,5 [-4,2; -1,0] | sim |
| DeBERTinha - TF-IDF + SVM | +2,6 [+1,1; +4,2] | +2,6 [+1,1; +4,2] | -2,8 [-4,5; -1,3] | sim |

**Transcrições da voz do autor** (184 itens)

| Comparação | Acurácia (p.p.) | F1 macro (p.p.) | EER (p.p.) | F1 significativo? |
|---|---|---|---|---|
| BERTimbau - TF-IDF + Regressão Logística | +1,0 [-6,1; +7,4] | +1,2 [-6,0; +8,0] | -5,2 [-14,1; +2,8] | não |
| BERTimbau - TF-IDF + SVM | +3,1 [-4,7; +10,7] | +3,3 [-4,8; +11,2] | -3,7 [-12,9; +4,6] | não |
| Albertina - TF-IDF + Regressão Logística | +1,5 [-5,3; +8,0] | +1,7 [-5,6; +8,6] | -4,5 [-14,0; +4,1] | não |
| Albertina - TF-IDF + SVM | +3,6 [-3,9; +11,2] | +3,8 [-4,1; +12,0] | -3,0 [-11,8; +5,9] | não |
| DeBERTinha - TF-IDF + Regressão Logística | +1,0 [-5,9; +7,4] | +1,1 [-6,1; +7,9] | -4,9 [-14,3; +4,7] | não |
| DeBERTinha - TF-IDF + SVM | +3,1 [-4,6; +10,7] | +3,3 [-4,6; +11,3] | -3,3 [-13,0; +6,2] | não |

Diferença A - B com IC 95% (bootstrap pareado, 2.000 reamostras). No EER, negativo favorece A. Na voz do autor, as mesmas frases foram lidas nos 4 cenários, então o bootstrap sorteia as frases (as leituras de cada uma entram juntas). Fontes: `reports/experimento/bootstrap.md` e `reports/asr_*/hipotese.md`.

## Tabela 5. Erros mais confiantes do BERTimbau no teste interno

| Erro | Frase (normalizada) | p(DD) do BERTimbau |
|---|---|---|
| Rotulada NDD, decidida DD | volta o video quero ver de novo aquela parte engracada | 0,9999 |
| Rotulada NDD, decidida DD | insere um sumario rapido antes da reuniao comecar | 0,9999 |
| Rotulada NDD, decidida DD | compra um mouse com taxa de polling ajustavel por software | 0,9999 |
| Rotulada NDD, decidida DD | cria esse atalho aqui na area de trabalho por favor | 0,9998 |
| Rotulada NDD, decidida DD | tira print dessa tela antes que a promocao suma amor | 0,9997 |
| Rotulada DD, decidida NDD | pesquiso as notas de corte aqui | 0,0001 |
| Rotulada DD, decidida NDD | ta tudo travado porque eu to tentando ditar e o sistema nao responde | 0,0001 |
| Rotulada DD, decidida NDD | nao quero ver essa bagunca so o drive | 0,0008 |
| Rotulada DD, decidida NDD | extrai os dados desse slide pra mim | 0,0010 |
| Rotulada DD, decidida NDD | vamos mudar o alvo para aquela coordenada por favor | 0,0011 |

Os 5 erros mais confiantes de cada tipo. O teste mede a concordância com o rótulo do LLM gerador, e um erro confiante pode ser um rótulo errado no corpus. Fonte: `reports/experimento/erros.md`.

## Tabela 6. Falsas ativações nas transcrições dos corpora públicos

| Modelo | CORAA | NURC-SP | TAGARELA | Geral |
|---|---|---|---|---|
| BERTimbau | 4,7% | 6,6% | 7,8% | 6,3% |
| Albertina | 3,5% | 5,3% | 7,5% | 5,4% |
| DeBERTinha | 5,1% | 5,9% | 6,0% | 5,7% |
| TF-IDF + Regressão Logística | 6,3% | 6,3% | 5,6% | 6,1% |
| TF-IDF + SVM | 6,8% | 6,6% | 6,4% | 6,6% |
| BERTimbau sem correção | 34,9% | 20,5% | 26,1% | 27,2% |
| Albertina sem correção | 28,1% | 16,2% | 21,7% | 22,0% |
| DeBERTinha sem correção | 41,4% | 23,9% | 26,8% | 30,8% |

| Modelo | s42 | s43 | s44 | Média ± desvio |
|---|---|---|---|---|
| BERTimbau | 5,0% | 5,7% | 6,3% | 5,7% ± 0,7 p.p. |
| Albertina | 5,4% | 4,9% | 6,2% | 5,5% ± 0,7 p.p. |
| DeBERTinha | 4,3% | 4,1% | 5,7% | 4,7% ± 0,9 p.p. |

2.912 transcrições dos próprios corpora (não do Whisper), todas não dirigidas: toda ativação é falsa. Fonte: `reports/externa_texto/resultado.md`.

## Tabela 7. Impacto do ASR

| Condição | Itens | WER médio |
|---|---|---|
| Voz sintética | 740 | 0,049 |
| Sala quieta | 40 | 0,137 |
| Ruído de fundo | 40 | 0,220 |
| iPhone | 40 | 0,205 |
| ~2 m | 40 | 0,201 |
| Roteiro de conversa | 24 | 0,031 |
| Áudio dos corpora públicos | 600 | 0,274 |

No áudio dos corpora públicos, sem as pausas preenchidas ("eh", "ahn"…) da referência e da transcrição, o WER médio fica em 0,271.

**Voz sintética** (740 itens)

| Modelo | Acerto no texto lido | Acerto na transcrição | Queda | Spearman (WER × queda) |
|---|---|---|---|---|
| BERTimbau | 97,8% | 97,4% | 0,4 p.p. | 0,09 |
| Albertina | 98,4% | 98,2% | 0,1 p.p. | 0,04 |
| DeBERTinha | 97,6% | 97,4% | 0,1 p.p. | 0,04 |
| TF-IDF + Regressão Logística | 96,2% | 94,9% | 1,4 p.p. | 0,20 |
| TF-IDF + SVM | 96,2% | 94,9% | 1,4 p.p. | 0,22 |

**Voz do autor** (184 itens)

| Modelo | Acerto no texto lido | Acerto na transcrição | Queda | Spearman (WER × queda) |
|---|---|---|---|---|
| BERTimbau | 95,7% | 93,5% | 2,2 p.p. | 0,06 |
| Albertina | 95,7% | 94,0% | 1,6 p.p. | 0,15 |
| DeBERTinha | 95,7% | 93,5% | 2,2 p.p. | 0,18 |
| TF-IDF + Regressão Logística | 97,3% | 92,4% | 4,9 p.p. | 0,27 |
| TF-IDF + SVM | 95,1% | 90,2% | 4,9 p.p. | 0,27 |

**Áudio dos corpora públicos** (600 trechos, todos não dirigidos)

| Modelo | Falsas ativações na transcrição do corpus | Falsas ativações na transcrição do Whisper | Diferença | Spearman (WER × queda) |
|---|---|---|---|---|
| BERTimbau | 5,3% | 5,8% | +0,5 p.p. | 0,05 |
| Albertina | 3,8% | 5,0% | +1,2 p.p. | 0,08 |
| DeBERTinha | 4,7% | 5,3% | +0,7 p.p. | 0,02 |
| TF-IDF + Regressão Logística | 5,7% | 7,0% | +1,3 p.p. | 0,05 |
| TF-IDF + SVM | 5,8% | 7,2% | +1,3 p.p. | 0,06 |
| BERTimbau sem correção | 21,0% | 22,3% | +1,3 p.p. | 0,03 |
| Albertina sem correção | 15,2% | 17,3% | +2,2 p.p. | 0,07 |
| DeBERTinha sem correção | 22,8% | 25,2% | +2,3 p.p. | 0,07 |

Nos três blocos, os mesmos itens nas duas versões do texto. No áudio público, a referência é a transcrição do próprio corpus, e o Whisper transcreve cada trecho inteiro, sem o detector de voz.

Fontes: `reports/asr_*/resumo.md` e `reports/asr_{tts,gravacoes}/hipotese.md`.

## Tabela 8. Sistema completo em tempo real

**Voz do autor** (transcrições idênticas entre as execuções: sim)

| Modelo | Pedidos atendidos (DD) | Falsas ativações (NDD) |
|---|---|---|
| BERTimbau | 95,0% | 7,7% |
| TF-IDF + Regressão Logística | 87,5% | 5,8% |
| TF-IDF + SVM | 87,5% | 8,7% |
| DeBERTinha | 92,5% | 5,8% |
| Albertina | 92,5% | 5,8% |
| BERTimbau sem correção | 92,5% | 26,0% |
| Albertina sem correção | 93,8% | 29,8% |
| DeBERTinha sem correção | 92,5% | 25,0% |

**Voz sintética** (transcrições idênticas entre as execuções: sim)

| Modelo | Pedidos atendidos (DD) | Falsas ativações (NDD) |
|---|---|---|
| BERTimbau | 96,5% | 1,9% |
| TF-IDF + Regressão Logística | 93,2% | 4,1% |
| TF-IDF + SVM | 94,3% | 5,1% |
| DeBERTinha | 95,9% | 1,9% |
| Albertina | 96,8% | 1,1% |
| BERTimbau sem correção | 94,1% | 34,3% |
| Albertina sem correção | 94,1% | 32,7% |
| DeBERTinha sem correção | 94,1% | 36,2% |

**Áudio dos corpora públicos** (transcrições idênticas entre as execuções: sim)

| Modelo | Pedidos atendidos (DD) | Falsas ativações (NDD) |
|---|---|---|
| BERTimbau | - | 6,5% |
| TF-IDF + Regressão Logística | - | 8,5% |
| TF-IDF + SVM | - | 8,7% |
| DeBERTinha | - | 5,3% |
| Albertina | - | 5,2% |
| BERTimbau sem correção | - | 29,2% |
| Albertina sem correção | - | 23,7% |
| DeBERTinha sem correção | - | 32,3% |

Os modelos com correção rodaram ao vivo, cada um na sua execução, com a semente fixa do Whisper; os sem correção julgaram as transcrições da execução do BERTimbau. As diferenças pareadas, com IC 95%, estão na figura R6 e em `reports/pipeline_*/comparacao.md`; na voz do autor, o bootstrap sorteia as frases. O áudio dos corpora públicos só tem falas não dirigidas, então mede apenas as falsas ativações.

## Tabela 9. Tempo de resposta do sistema completo

| Conjunto | Falas | Mediana | Percentil 95 | Com nova tentativa do Whisper | Percentil 95 sem elas | Whisper (mediana) | Classificador (mediana) |
|---|---|---|---|---|---|---|---|
| Voz do autor | 198 | 1,18 s | 1,36 s | 3 | 1,31 s | 0,43 s | 41 ms |
| Voz sintética | 751 | 1,17 s | 1,27 s | 0 | 1,27 s | 0,42 s | 38 ms |
| Áudio dos corpora públicos | 691 | 1,15 s | 1,91 s | 39 | 1,25 s | 0,42 s | 28 ms |

Do fim da fala até a decisão, com a espera de 0,7 s de silêncio do detector de voz. Execução do BERTimbau. O resto (filas e processamento do detector de voz) tem percentil 95 de 14 a 16 ms nas falas fechadas pelo fim da voz. Fonte: `reports/pipeline_*/bertimbau/`.

**Whisper na GPU × na CPU**

| Whisper large-v3-turbo | Latência (mediana) | RTF |
|---|---|---|
| mlx-whisper fp16, GPU (o do sistema) | 0,45 s | 0,090 |
| faster-whisper fp32, CPU, 4 threads | 2,93 s | 0,586 |
| faster-whisper int8, CPU, 4 threads | 5,23 s | 1,045 |

Benchmark com 5 s de fala sintetizada pelo macOS (RTF = tempo de processamento ÷ duração do áudio), na tomada, com carga do sistema de 1,4 no início. Mede só o tempo, não a qualidade da transcrição. Fonte: `reports/benchmark/benchmark-20261105-1623.json`.

## Tabela 10. Custo dos classificadores

| Modelo | Parâmetros | Tamanho | Treino por semente | Pico de memória | Inferência ao vivo (CPU) |
|---|---|---|---|---|---|
| BERTimbau | 108,9 mi | 416 MB | 56 ± 5 min | 3,4 GB | 27,7-41,1 ms |
| Albertina | 139,2 mi | 531 MB | 148 ± 3 min | 8,8 GB | 43,4-44,1 ms |
| DeBERTinha | 40,9 mi | 156 MB | 50 ± 3 min | 2,4 GB | 31,1-31,3 ms |
| TF-IDF + Regressão Logística | 50.000 termos | 0,9 MB | ~5 s | - | 0,64-1,41 ms |
| TF-IDF + SVM | 50.000 termos | 2,3 MB | ~4 s | - | 4,17-4,27 ms |

Parâmetros contados nos pesos do modelo treinado. O "100m" do nome da Albertina conta o codificador sem a camada de embeddings (99,8 milhões); com ela, são 139,2 milhões. Inferência: mediana por fala na CPU, com as threads do sistema, medida no sistema ao vivo; a faixa vai do menor ao maior valor nos 3 conjuntos (cada modelo na sua execução). A do BERTimbau variou entre as sessões sem causa conhecida. Fontes: `reports/custo/custo.md` e `reports/pipeline_*/comparacao.json`.

## Tabela 11. Corpus final

| Origem | Modelo gerador | Estratégia | Frases por classe | Parte do corpus |
|---|---|---|---|---|
| Corpus inicial | qwen/qwen3.6-35b-a3b | - | 20.415 | 35,1% |
| Corpus inicial | claude-sonnet (CLI) | - | 10.390 | 17,8% |
| Corpus inicial | openrouter/free | - | 3.369 | 5,8% |
| Corpus inicial | qwen/qwen3.8-27b | - | 3.088 | 5,3% |
| Corpus inicial | zai-org/glm-4.6v-flash | - | 2.608 | 4,5% |
| Corpus inicial | opencode (CLI) | - | 2.342 | 4,0% |
| Corpus inicial | google/gemma-4-e4b | - | 1.973 | 3,4% |
| Corpus inicial | mistral-small-latest | - | 1.829 | 3,1% |
| Corpus inicial | moonshotai/kimi-k3 | - | 1.681 | 2,9% |
| Frases novas da correção | google/gemma-4-e4b | por tamanho | 3.982 | 6,8% |
| Frases novas da correção | google/gemma-4-e4b | negativos difíceis | 1.425 | 2,4% |
| Frases novas da correção | claude-sonnet-5-5 | negativos difíceis | 1.211 | 2,1% |
| Frases novas da correção | qwen/qwen3.8-27b | negativos difíceis | 985 | 1,7% |
| Frases novas da correção | openai/gpt-oss-120b | negativos difíceis | 916 | 1,6% |
| Frases novas da correção | qwen/qwen3.8-27b | por tamanho | 914 | 1,6% |
| Frases novas da correção | openai/gpt-oss-120b | por tamanho | 703 | 1,2% |
| Frases novas da correção | claude-sonnet-5-5 | por tamanho | 361 | 0,6% |
| Frases novas da correção | gemini-3.5-flash | por tamanho | 27 | < 0,1% |
| **Corpus inicial** |  |  | **47.695** | **81,9%** |
| **Frases novas da correção** |  |  | **10.524** | **18,1%** |

| Partição | Frases por classe | Total |
|---|---|---|
| Treino | 46.575 | 93.150 |
| Validação | 5.836 | 11.672 |
| Teste | 5.808 | 11.616 |

Corpus final com 116.438 frases, metade de cada classe em cada linha. Mediana de 9 palavras nas duas classes (2 a 24). Só o tamanho previa a classe com AUC de 0,818 depois dos filtros; no corpus final, 0,515. As partições não têm texto em comum. Fonte: `reports/corpus/principal/composicao.md`.

## Tabela 12. Treino dos classificadores

| Modelo | Semente | Melhor checkpoint (época) | F1 macro de validação | Limiar escolhido | No sistema |
|---|---|---|---|---|---|
| BERTimbau | 42 | 4 de 4 | 0,9830 | 0,420 |  |
| BERTimbau | 43 | 4 de 4 | 0,9833 | 0,725 |  |
| BERTimbau | 44 | 3 de 4 | 0,9834 | 0,685 | sim |
| Albertina | 42 | 4 de 4 | 0,9792 | 0,620 | sim |
| Albertina | 43 | 4 de 4 | 0,9787 | 0,940 |  |
| Albertina | 44 | 4 de 4 | 0,9789 | 0,555 |  |
| DeBERTinha | 42 | 5 de 5 | 0,9787 | 0,830 |  |
| DeBERTinha | 43 | 5 de 5 | 0,9793 | 0,800 |  |
| DeBERTinha | 44 | 5 de 5 | 0,9798 | 0,395 | sim |
| BERTimbau sem correção | 42 | 4 de 4 | 0,9887 | 0,070 |  |
| Albertina sem correção | 42 | 4 de 4 | 0,9879 | 0,490 |  |
| DeBERTinha sem correção | 42 | 5 de 5 | 0,9874 | 0,030 |  |

F1 macro de validação do modelo final, com limiar 0,5. O melhor checkpoint é o de maior F1 macro de validação; o limiar maximiza a mesma F1 e varia muito entre sementes (0,395 a 0,940 com correção e 0,030 a 0,490 sem correção; a tabela 13 mostra o efeito). Em 11 de 12 execuções, a menor perda de validação vem antes do melhor checkpoint: depois dela, a perda sobe, mas a F1 ainda melhora. A validação dos modelos sem correção é a do corpus deles. TF-IDF (C por validação cruzada de 5 partes): TF-IDF + Regressão Logística: C = 10, F1 macro de validação 0,9605 e limiar 0,48; TF-IDF + SVM: C = 1, F1 macro de validação 0,9611 e limiar 0,46. Fonte: `reports/treino/treino.md`.

**Hiperparâmetros**

| Hiperparâmetro | BERTimbau | Albertina | DeBERTinha |
|---|---|---|---|
| Taxa de aprendizado | 2e-5 | 2e-5 | 2e-5 |
| Lote | 32 | 32 | 32 |
| Épocas (máximo) | 4 | 4 | 5 |
| Tokens por frase (máximo) | 128 | 128 | 128 |
| Aquecimento | 10% | 10% | 10% |
| Dropout | 0,1 | 0,1 | 0,1 |
| Weight decay | 0,01 | 0,01 | 0,01 |
| Parada antecipada (paciência, em épocas) | 2 | 2 | 2 |
| Precisão | bf16 | fp32 | bf16 |

## Tabela 13. Falsas ativações com o limiar da validação e com 0,5

| Modelo | Limiar da validação | Voz do autor | Voz sintética | Áudio dos corpora públicos |
|---|---|---|---|---|
| BERTimbau | 0,685 | 7,7% → 8,7% | 1,9% → 2,2% | 6,5% → 6,5% |
| Albertina | 0,620 | 5,8% → 5,8% | 1,1% → 1,1% | 5,2% → 5,5% |
| DeBERTinha | 0,395 | 5,8% → 5,8% | 1,9% → 1,9% | 5,3% → 5,2% |
| BERTimbau sem correção | 0,070 | 26,0% → 26,0% | 34,3% → 33,0% | 29,2% → 27,2% |
| Albertina sem correção | 0,490 | 29,8% → 29,8% | 32,7% → 32,7% | 23,7% → 23,7% |
| DeBERTinha sem correção | 0,030 | 25,0% → 25,0% | 36,2% → 33,2% | 32,3% → 26,8% |

Falsas ativações no sistema completo com o limiar escolhido na validação → com 0,5. Com 0,5 em todos, os modelos sem correção ficam em 23,7%-33,2%, contra 1,1%-8,7% com correção. Os p(DD) gravados em cada fala reproduzem as decisões do sistema. Fonte: `reports/pipeline_*/*/registros.jsonl`.
