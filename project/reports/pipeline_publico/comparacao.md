# Comparação dos classificadores no sistema completo - publico

Os modelos com correção rodaram ao vivo, cada um na sua execução, com a semente fixa do Whisper; os sem correção julgaram as transcrições da execução do BERTimbau. Ativação = fração de itens decididos DD: em DD são os pedidos atendidos; nos demais, falsas ativações.

| execução | itens | mesmos trechos | mesma transcrição |
|---|---|---|---|
| TF-IDF + Regressão Logística | 600 | 600 | 600 |
| TF-IDF + SVM | 600 | 600 | 600 |
| DeBERTinha | 600 | 600 | 600 |
| Albertina | 600 | 600 | 600 |

Paridade dos modelos sem correção: o BERTimbau, julgando de novo, reproduziu 691 de 691 decisões da execução em tempo real.

| grupo | rótulo | n | BERTimbau | TF-IDF + Regressão Logística | TF-IDF + SVM | DeBERTinha | Albertina | BERTimbau sem correção | DeBERTinha sem correção | Albertina sem correção |
|---|---|---|---|---|---|---|---|---|---|---|
| coraa | NDD | 300 | 0.0733 | 0.0967 | 0.1033 | 0.0500 | 0.0467 | 0.3467 | 0.4100 | 0.2933 |
| nurcsp | NDD | 300 | 0.0567 | 0.0733 | 0.0700 | 0.0567 | 0.0567 | 0.2367 | 0.2367 | 0.1800 |

### Diferenças pareadas (A - B; todos os grupos juntos; bootstrap de 2.000)

Cada modelo contra o BERTimbau e cada Transformer contra a própria versão sem correção.

| comparação | rótulo | ativação A | ativação B | diferença | IC95 inf | IC95 sup |
|---|---|---|---|---|---|---|
| TF-IDF + Regressão Logística - BERTimbau | NDD | 0.0850 | 0.0650 | 0.0200 | -0.0050 | 0.0450 |
| TF-IDF + SVM - BERTimbau | NDD | 0.0867 | 0.0650 | 0.0217 | -0.0034 | 0.0483 |
| DeBERTinha - BERTimbau | NDD | 0.0533 | 0.0650 | -0.0117 | -0.0300 | 0.0067 |
| Albertina - BERTimbau | NDD | 0.0517 | 0.0650 | -0.0133 | -0.0333 | 0.0067 |
| BERTimbau sem correção - BERTimbau | NDD | 0.2917 | 0.0650 | 0.2267 | 0.1917 | 0.2617 |
| DeBERTinha sem correção - DeBERTinha | NDD | 0.3233 | 0.0533 | 0.2700 | 0.2333 | 0.3067 |
| Albertina sem correção - Albertina | NDD | 0.2367 | 0.0517 | 0.1850 | 0.1533 | 0.2183 |

### Tempo de inferência por fala no sistema ao vivo (CPU, mesmas threads)

Cada modelo medido na sua execução ao vivo. Os modelos sem correção têm a mesma arquitetura dos com correção e, portanto, o mesmo custo.

| classificador | falas | p50 (ms) | p95 (ms) |
|---|---|---|---|
| BERTimbau | 689 | 27.7400 | 32.3120 |
| TF-IDF + Regressão Logística | 689 | 0.6400 | 1.7800 |
| TF-IDF + SVM | 689 | 4.2500 | 5.2600 |
| DeBERTinha | 689 | 31.3100 | 34.0100 |
| Albertina | 689 | 44.1000 | 53.3760 |
