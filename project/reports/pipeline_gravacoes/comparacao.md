# Comparação dos classificadores no sistema completo - gravacoes

Os modelos com correção rodaram ao vivo, cada um na sua execução, com a semente fixa do Whisper; os sem correção julgaram as transcrições da execução do BERTimbau. Ativação = fração de itens decididos DD: em DD são os pedidos atendidos; nos demais, falsas ativações.

| execução | itens | mesmos trechos | mesma transcrição |
|---|---|---|---|
| TF-IDF + Regressão Logística | 192 | 192 | 192 |
| TF-IDF + SVM | 192 | 192 | 192 |
| DeBERTinha | 192 | 192 | 192 |
| Albertina | 192 | 192 | 192 |

Paridade dos modelos sem correção: o BERTimbau, julgando de novo, reproduziu 198 de 198 decisões da execução em tempo real.

| grupo | rótulo | n | BERTimbau | TF-IDF + Regressão Logística | TF-IDF + SVM | DeBERTinha | Albertina | BERTimbau sem correção | DeBERTinha sem correção | Albertina sem correção |
|---|---|---|---|---|---|---|---|---|---|---|
| conversa_interno | NDD | 24 | 0.0000 | 0.0417 | 0.0417 | 0.0000 | 0.0000 | 0.2083 | 0.2083 | 0.2917 |
| distante_interno | DD | 20 | 0.9500 | 0.8500 | 0.8500 | 0.9500 | 0.9500 | 0.9500 | 0.9500 | 0.9000 |
| distante_interno | NDD | 20 | 0.1500 | 0.1500 | 0.2000 | 0.1500 | 0.1500 | 0.4000 | 0.3500 | 0.4000 |
| distante_interno | NDD_LIVRE | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| distante_interno | RUIDO | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| quieto_interno | DD | 20 | 0.9500 | 0.9000 | 0.9000 | 0.9000 | 0.9000 | 0.9000 | 0.9000 | 0.9500 |
| quieto_interno | NDD | 20 | 0.1000 | 0.0000 | 0.0000 | 0.0500 | 0.0500 | 0.2500 | 0.2500 | 0.2500 |
| quieto_interno | NDD_LIVRE | 1 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| quieto_interno | RUIDO | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| quieto_iphone | DD | 20 | 0.9500 | 0.9000 | 0.9000 | 0.9000 | 0.9000 | 0.9500 | 0.9500 | 0.9500 |
| quieto_iphone | NDD | 20 | 0.1000 | 0.0500 | 0.1000 | 0.0500 | 0.0500 | 0.2000 | 0.2000 | 0.2500 |
| quieto_iphone | NDD_LIVRE | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| quieto_iphone | RUIDO | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ruido_interno | DD | 20 | 0.9500 | 0.8500 | 0.8500 | 0.9500 | 0.9500 | 0.9000 | 0.9000 | 0.9500 |
| ruido_interno | NDD | 20 | 0.0500 | 0.0500 | 0.1000 | 0.0500 | 0.0500 | 0.2500 | 0.2500 | 0.3000 |
| ruido_interno | NDD_LIVRE | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ruido_interno | RUIDO | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### Diferenças pareadas (A - B; todos os grupos juntos; bootstrap de 2.000)

Cada modelo contra o BERTimbau e cada Transformer contra a própria versão sem correção. O bootstrap sorteia as frases: as leituras da mesma frase nos 4 cenários entram juntas, porque os erros de um modelo se repetem na mesma frase.

| comparação | rótulo | ativação A | ativação B | diferença | IC95 inf | IC95 sup |
|---|---|---|---|---|---|---|
| TF-IDF + Regressão Logística - BERTimbau | DD | 0.8750 | 0.9500 | -0.0750 | -0.1625 | -0.0125 |
| TF-IDF + Regressão Logística - BERTimbau | NDD | 0.0577 | 0.0769 | -0.0192 | -0.1122 | 0.0632 |
| TF-IDF + SVM - BERTimbau | DD | 0.8750 | 0.9500 | -0.0750 | -0.1625 | -0.0125 |
| TF-IDF + SVM - BERTimbau | NDD | 0.0865 | 0.0769 | 0.0096 | -0.0962 | 0.1151 |
| DeBERTinha - BERTimbau | DD | 0.9250 | 0.9500 | -0.0250 | -0.0750 | 0.0000 |
| DeBERTinha - BERTimbau | NDD | 0.0577 | 0.0769 | -0.0192 | -0.0843 | 0.0240 |
| Albertina - BERTimbau | DD | 0.9250 | 0.9500 | -0.0250 | -0.0750 | 0.0000 |
| Albertina - BERTimbau | NDD | 0.0577 | 0.0769 | -0.0192 | -0.0612 | 0.0000 |
| BERTimbau sem correção - BERTimbau | DD | 0.9250 | 0.9500 | -0.0250 | -0.1625 | 0.1000 |
| BERTimbau sem correção - BERTimbau | NDD | 0.2596 | 0.0769 | 0.1827 | 0.0096 | 0.3478 |
| DeBERTinha sem correção - DeBERTinha | DD | 0.9250 | 0.9250 | 0.0000 | -0.1500 | 0.1375 |
| DeBERTinha sem correção - DeBERTinha | NDD | 0.2500 | 0.0577 | 0.1923 | 0.0467 | 0.3374 |
| Albertina sem correção - Albertina | DD | 0.9375 | 0.9250 | 0.0125 | -0.1375 | 0.1500 |
| Albertina sem correção - Albertina | NDD | 0.2981 | 0.0577 | 0.2404 | 0.0796 | 0.4045 |

### De onde vêm os erros

Segmentação: o detector de voz não achou fala ou dividiu o item. Classificador: o modelo também erra no texto lido. Transcrição: acerta no texto lido e erra na transcrição.

| modelo | rótulo | segmentação | classificador | transcrição |
|---|---|---|---|---|
| BERTimbau | DD | 0 | 3 | 1 |
| BERTimbau | NDD | 1 | 3 | 4 |
| TF-IDF + Regressão Logística | DD | 0 | 4 | 6 |
| TF-IDF + Regressão Logística | NDD | 1 | 1 | 4 |
| TF-IDF + SVM | DD | 0 | 4 | 6 |
| TF-IDF + SVM | NDD | 1 | 4 | 4 |
| DeBERTinha | DD | 0 | 3 | 3 |
| DeBERTinha | NDD | 1 | 3 | 2 |
| Albertina | DD | 0 | 3 | 3 |
| Albertina | NDD | 1 | 3 | 2 |
| BERTimbau sem correção | DD | 0 | 6 | 0 |
| BERTimbau sem correção | NDD | 3 | 20 | 4 |
| DeBERTinha sem correção | DD | 0 | 6 | 0 |
| DeBERTinha sem correção | NDD | 3 | 20 | 3 |
| Albertina sem correção | DD | 0 | 4 | 1 |
| Albertina sem correção | NDD | 3 | 21 | 7 |

### Tempo de inferência por fala no sistema ao vivo (CPU, mesmas threads)

Cada modelo medido na sua execução ao vivo. Os modelos sem correção têm a mesma arquitetura dos com correção e, portanto, o mesmo custo.

| classificador | falas | p50 (ms) | p95 (ms) |
|---|---|---|---|
| BERTimbau | 198 | 41.0750 | 53.1300 |
| TF-IDF + Regressão Logística | 198 | 1.1100 | 1.7900 |
| TF-IDF + SVM | 198 | 4.1700 | 5.5515 |
| DeBERTinha | 198 | 31.0550 | 33.6910 |
| Albertina | 198 | 43.4600 | 53.0205 |
