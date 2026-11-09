# Nota sobre estas execuções (2026-11-04, 19h14-20h49, e 2026-11-05, 14h18-14h42)

- **O que é:** o sistema completo em tempo real, com os 5 modelos com correção, cada um na sua execução e no
  seu processo. Os 3 modelos sem correção julgaram as transcrições da execução do BERTimbau.
- **Dados:** a voz do autor, 192 itens tocados em tempo real no buffer do microfone.
  - Em cada um dos 4 cenários (sala quieta, ruído de fundo, iPhone e ~2 m): as mesmas 40 frases (20 DD e 20
    NDD), a fala livre e os 30 s de ambiente.
  - As 24 falas NDD do roteiro de conversa (`config/roteiro_conversa.txt`).
- **Mesmo áudio e mesma transcrição para todos:** semente fixa do Whisper (`[asr] seed = 0`), a mesma ordem
  dos itens e um processo por modelo.
  - Nas 5 execuções, 192 de 192 itens tiveram os mesmos trechos e a mesma transcrição.
  - Paridade: o BERTimbau, julgando de novo as transcrições, reproduziu 198 de 198 decisões.
- **Código:** o mesmo sistema ao vivo nas 5 execuções.
  - Regressão Logística, SVM, DeBERTinha e Albertina rodaram numa versão; o BERTimbau e a comparação, numa
    versão seguinte. Entre as duas, só mudaram os comandos de análise e de relatório.
- **Energia:** todas na tomada, sem Modo de Pouca Energia.
  - Regressão Logística, SVM e DeBERTinha rodaram com a bateria carregando.
  - Albertina e BERTimbau rodaram com 100%.
- **Logs:** sem avisos nem erros (`data/logs/pipeline_gravacoes_<modelo>.log` e
  `data/logs/comparar_gravacoes.log`).

## Resultado

| Modelo | Pedidos atendidos (DD) | Falsas ativações (NDD) | Roteiro de conversa | Fala livre |
|---|---|---|---|---|
| BERTimbau | 76/80 (95,0%) | 8/104 (7,7%) | 0/24 | 0/4 |
| TF-IDF + Regressão Logística | 70/80 (87,5%) | 6/104 (5,8%) | 1/24 | 1/4 |
| TF-IDF + SVM | 70/80 (87,5%) | 9/104 (8,7%) | 1/24 | 1/4 |
| DeBERTinha | 74/80 (92,5%) | 6/104 (5,8%) | 0/24 | 0/4 |
| Albertina | 74/80 (92,5%) | 6/104 (5,8%) | 0/24 | 0/4 |
| BERTimbau sem correção | 74/80 (92,5%) | 27/104 (26,0%) | 5/24 | 1/4 |
| DeBERTinha sem correção | 74/80 (92,5%) | 26/104 (25,0%) | 5/24 | 1/4 |
| Albertina sem correção | 75/80 (93,8%) | 31/104 (29,8%) | 7/24 | 1/4 |

- **NDD:** as 80 frases dos cenários mais as 24 falas do roteiro de conversa.
- **Fala livre:** arquivos com pelo menos uma ativação.
- **Só o ambiente:** em nenhum cenário os 30 s abriram uma fala no detector de voz.
- **Por cenário:** a tabela completa está em `comparacao.md`.

## Diferenças pareadas (bootstrap por frase)

As mesmas frases foram lidas nos 4 cenários, e os erros se repetem na mesma frase. Por exemplo, "passa pro lado
direito" errou em 3 ou 4 cenários em todos os modelos. Por isso o bootstrap sorteia as frases (20 DD e 44 NDD),
e as leituras de cada frase entram juntas. Os intervalos ficam mais largos que no sorteio por item.

| Comparação | Diferença | IC 95% |
|---|---|---|
| Sem correção × com correção, falsas ativações: BERTimbau | +18,3 p.p. | [+1,0; +34,8] |
| Sem correção × com correção, falsas ativações: DeBERTinha | +19,2 p.p. | [+4,7; +33,7] |
| Sem correção × com correção, falsas ativações: Albertina | +24,0 p.p. | [+8,0; +40,5] |
| TF-IDF (os dois) × BERTimbau, pedidos atendidos | -7,5 p.p. | [-16,3; -1,3] |
| TF-IDF × BERTimbau, falsas ativações | -1,9 e +1,0 p.p. | incluem o zero |
| DeBERTinha e Albertina × BERTimbau | -2,5 p.p. (DD) e -1,9 p.p. (NDD) | incluem o zero |

- **Sem correção:** os pedidos atendidos não mudam de forma significativa (de -2,5 a +1,3 p.p.).
- **TF-IDF:** os 10 pedidos perdidos estão em 5 frases distintas, e 6 deles vêm da transcrição (o modelo
  acerta no texto lido). No BERTimbau, 1 dos 4 vem da transcrição.

## De onde vêm os erros

| Modelo | DD: classificador / transcrição | NDD: segmentação / classificador / transcrição |
|---|---|---|
| BERTimbau | 3 / 1 | 1 / 3 / 4 |
| TF-IDF + Regressão Logística | 4 / 6 | 1 / 1 / 4 |
| TF-IDF + SVM | 4 / 6 | 1 / 4 / 4 |
| DeBERTinha | 3 / 3 | 1 / 3 / 2 |
| Albertina | 3 / 3 | 1 / 3 / 2 |
| BERTimbau sem correção | 6 / 0 | 3 / 20 / 4 |
| DeBERTinha sem correção | 6 / 0 | 3 / 20 / 3 |
| Albertina sem correção | 4 / 1 | 3 / 21 / 7 |

- **Classificador:** o modelo também erra no texto lido.
- **Transcrição:** o modelo acerta no texto lido e erra na transcrição.
- **Segmentação:** o detector de voz dividiu o item. Os trechos são os mesmos para todos os modelos; muda só
  em quantos desses itens cada modelo errou.
- **Nenhum pedido DD se perdeu na segmentação.**

## Latência (do fim da fala até a decisão, com os 0,7 s de espera do detector de voz)

| Execução | Whisper p50 | Total p50 | Total p95 | Classificador p50 |
|---|---|---|---|---|
| BERTimbau | 0,43 s | 1,18 s | 1,36 s | 41 ms |
| TF-IDF + Regressão Logística | 0,43 s | 1,14 s | 1,31 s | 1,1 ms |
| TF-IDF + SVM | 0,46 s | 1,18 s | 1,29 s | 4,2 ms |
| DeBERTinha | 0,42 s | 1,16 s | 1,30 s | 31 ms |
| Albertina | 0,42 s | 1,17 s | 1,30 s | 43 ms |

- **198 falas por execução.**
- **Whisper:** o mesmo em todas, com o mesmo áudio. Entre as execuções, a mediana dele variou de 0,42 a
  0,46 s.
- **Classificador do BERTimbau:** 41 ms nesta execução. Entre as sessões, ele variou de 28 a 41 ms (38 ms na
  voz sintética e 28 ms no áudio público), sem causa conhecida. DeBERTinha e Albertina ficaram iguais nos três
  conjuntos.
- **Classificador:** de 1 a 43 ms, até 4% do total. O custo dos modelos sem correção é o mesmo dos com
  correção, porque a arquitetura é a mesma.
