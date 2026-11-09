# Impacto do ASR - tts (740 itens, backend mlx)

### WER médio 0.04876372023220522

Spearman(confiabilidade fuzzy, WER) = -0.029 (p = 0.42)

Referência e ASR nos mesmos 740 itens (pareado).

| modelo | acc referência | acc ASR | FP ASR | FN ASR |
|---|---|---|---|---|
| debertinha-s44 | 0.9757 | 0.9743 | 0.0189 | 0.0324 |
| controle/debertinha-s42 | 0.7946 | 0.7959 | 0.3514 | 0.0568 |
| bertimbau-s44 | 0.9784 | 0.9743 | 0.0189 | 0.0324 |
| controle/bertimbau-s42 | 0.8000 | 0.8014 | 0.3378 | 0.0595 |
| albertina-s42 | 0.9838 | 0.9824 | 0.0108 | 0.0243 |
| controle/albertina-s42 | 0.8081 | 0.8122 | 0.3216 | 0.0541 |
| baseline_logreg | 0.9622 | 0.9486 | 0.0351 | 0.0676 |
| baseline_svm | 0.9622 | 0.9486 | 0.0459 | 0.0568 |
