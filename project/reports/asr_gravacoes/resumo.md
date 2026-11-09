# Impacto do ASR - gravacoes (188 itens, backend mlx)

### WER médio 0.16980757689424736

Spearman(confiabilidade fuzzy, WER) = -0.291 (p = 6.1e-05)

Referência e ASR nos mesmos 184 itens (pareado). Fala livre sem roteiro (4 itens) à parte: só FP.

| modelo | acc referência | acc ASR | FP ASR | FN ASR | FP fala livre |
|---|---|---|---|---|---|
| debertinha-s44 | 0.9565 | 0.9348 | 0.0481 | 0.0875 | 0.0000 |
| controle/debertinha-s42 | 0.8478 | 0.8261 | 0.2308 | 0.1000 | 0.0000 |
| bertimbau-s44 | 0.9565 | 0.9348 | 0.0673 | 0.0625 | 0.0000 |
| controle/bertimbau-s42 | 0.8478 | 0.8315 | 0.2212 | 0.1000 | 0.0000 |
| albertina-s42 | 0.9565 | 0.9402 | 0.0385 | 0.0875 | 0.0000 |
| controle/albertina-s42 | 0.8641 | 0.8207 | 0.2500 | 0.0875 | 0.0000 |
| baseline_logreg | 0.9728 | 0.9239 | 0.0288 | 0.1375 | 0.0000 |
| baseline_svm | 0.9511 | 0.9022 | 0.0673 | 0.1375 | 0.0000 |
