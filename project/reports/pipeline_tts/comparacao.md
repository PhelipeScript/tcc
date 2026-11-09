# Comparação dos classificadores no sistema completo - tts

Os modelos com correção rodaram ao vivo, cada um na sua execução, com a semente fixa do Whisper; os sem correção julgaram as transcrições da execução do BERTimbau. Ativação = fração de itens decididos DD: em DD são os pedidos atendidos; nos demais, falsas ativações.

| execução | itens | mesmos trechos | mesma transcrição |
|---|---|---|---|
| TF-IDF + Regressão Logística | 740 | 740 | 740 |
| TF-IDF + SVM | 740 | 740 | 740 |
| DeBERTinha | 740 | 740 | 740 |
| Albertina | 740 | 740 | 740 |

Paridade dos modelos sem correção: o BERTimbau, julgando de novo, reproduziu 751 de 751 decisões da execução em tempo real.

| grupo | rótulo | n | BERTimbau | TF-IDF + Regressão Logística | TF-IDF + SVM | DeBERTinha | Albertina | BERTimbau sem correção | DeBERTinha sem correção | Albertina sem correção |
|---|---|---|---|---|---|---|---|---|---|---|
| tts:pt-BR-AntonioNeural | DD | 109 | 0.9725 | 0.9358 | 0.9541 | 0.9725 | 0.9908 | 0.9541 | 0.9541 | 0.9541 |
| tts:pt-BR-AntonioNeural | NDD | 138 | 0.0145 | 0.0217 | 0.0362 | 0.0145 | 0.0072 | 0.3623 | 0.3913 | 0.3478 |
| tts:pt-BR-FranciscaNeural | DD | 123 | 0.9431 | 0.9268 | 0.9268 | 0.9268 | 0.9512 | 0.9350 | 0.9431 | 0.9268 |
| tts:pt-BR-FranciscaNeural | NDD | 124 | 0.0081 | 0.0484 | 0.0484 | 0.0323 | 0.0161 | 0.3790 | 0.3871 | 0.3548 |
| tts:pt-BR-ThalitaMultilingualNeural | DD | 138 | 0.9783 | 0.9348 | 0.9493 | 0.9783 | 0.9638 | 0.9348 | 0.9275 | 0.9420 |
| tts:pt-BR-ThalitaMultilingualNeural | NDD | 108 | 0.0370 | 0.0556 | 0.0741 | 0.0093 | 0.0093 | 0.2778 | 0.2963 | 0.2685 |

### Diferenças pareadas (A - B; todos os grupos juntos; bootstrap de 2.000)

Cada modelo contra o BERTimbau e cada Transformer contra a própria versão sem correção.

| comparação | rótulo | ativação A | ativação B | diferença | IC95 inf | IC95 sup |
|---|---|---|---|---|---|---|
| TF-IDF + Regressão Logística - BERTimbau | DD | 0.9324 | 0.9649 | -0.0324 | -0.0595 | -0.0054 |
| TF-IDF + Regressão Logística - BERTimbau | NDD | 0.0405 | 0.0189 | 0.0216 | 0.0000 | 0.0459 |
| TF-IDF + SVM - BERTimbau | DD | 0.9432 | 0.9649 | -0.0216 | -0.0459 | 0.0027 |
| TF-IDF + SVM - BERTimbau | NDD | 0.0514 | 0.0189 | 0.0324 | 0.0081 | 0.0595 |
| DeBERTinha - BERTimbau | DD | 0.9595 | 0.9649 | -0.0054 | -0.0216 | 0.0081 |
| DeBERTinha - BERTimbau | NDD | 0.0189 | 0.0189 | 0.0000 | -0.0162 | 0.0162 |
| Albertina - BERTimbau | DD | 0.9676 | 0.9649 | 0.0027 | -0.0135 | 0.0189 |
| Albertina - BERTimbau | NDD | 0.0108 | 0.0189 | -0.0081 | -0.0216 | 0.0027 |
| BERTimbau sem correção - BERTimbau | DD | 0.9405 | 0.9649 | -0.0243 | -0.0514 | 0.0054 |
| BERTimbau sem correção - BERTimbau | NDD | 0.3432 | 0.0189 | 0.3243 | 0.2757 | 0.3730 |
| DeBERTinha sem correção - DeBERTinha | DD | 0.9405 | 0.9595 | -0.0189 | -0.0486 | 0.0108 |
| DeBERTinha sem correção - DeBERTinha | NDD | 0.3622 | 0.0189 | 0.3432 | 0.2946 | 0.3946 |
| Albertina sem correção - Albertina | DD | 0.9405 | 0.9676 | -0.0270 | -0.0541 | 0.0000 |
| Albertina sem correção - Albertina | NDD | 0.3270 | 0.0108 | 0.3162 | 0.2676 | 0.3649 |

### De onde vêm os erros

Segmentação: o detector de voz não achou fala ou dividiu o item. Classificador: o modelo também erra no texto lido. Transcrição: acerta no texto lido e erra na transcrição.

| modelo | rótulo | segmentação | classificador | transcrição |
|---|---|---|---|---|
| BERTimbau | DD | 1 | 9 | 3 |
| BERTimbau | NDD | 1 | 3 | 3 |
| TF-IDF + Regressão Logística | DD | 0 | 16 | 9 |
| TF-IDF + Regressão Logística | NDD | 2 | 11 | 2 |
| TF-IDF + SVM | DD | 0 | 12 | 9 |
| TF-IDF + SVM | NDD | 2 | 15 | 2 |
| DeBERTinha | DD | 1 | 11 | 3 |
| DeBERTinha | NDD | 0 | 6 | 1 |
| Albertina | DD | 0 | 8 | 4 |
| Albertina | NDD | 0 | 4 | 0 |
| BERTimbau sem correção | DD | 0 | 21 | 1 |
| BERTimbau sem correção | NDD | 3 | 123 | 1 |
| DeBERTinha sem correção | DD | 0 | 20 | 2 |
| DeBERTinha sem correção | NDD | 3 | 130 | 1 |
| Albertina sem correção | DD | 0 | 20 | 2 |
| Albertina sem correção | NDD | 2 | 119 | 0 |

### Tempo de inferência por fala no sistema ao vivo (CPU, mesmas threads)

Cada modelo medido na sua execução ao vivo. Os modelos sem correção têm a mesma arquitetura dos com correção e, portanto, o mesmo custo.

| classificador | falas | p50 (ms) | p95 (ms) |
|---|---|---|---|
| BERTimbau | 751 | 38.2300 | 46.3750 |
| TF-IDF + Regressão Logística | 751 | 1.4100 | 1.8600 |
| TF-IDF + SVM | 751 | 4.2700 | 5.2350 |
| DeBERTinha | 751 | 31.1400 | 33.3050 |
| Albertina | 751 | 43.3800 | 51.8100 |
