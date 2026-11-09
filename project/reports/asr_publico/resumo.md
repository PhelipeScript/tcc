# Impacto do ASR - publico (600 itens, backend mlx)

### WER médio 0.27405980673695

WER médio sem as pausas preenchidas (ah, ahm, ahn, eh, ehm, hm, hum, uh, uhm), removidas da referência e da transcrição: 0.2712469517368527

Spearman(confiabilidade fuzzy, WER) = -0.515 (p = 6.3e-42)

Referência e ASR nos mesmos 600 itens (pareado).

| modelo | acc referência | acc ASR | FP referência | FP ASR |
|---|---|---|---|---|
| debertinha-s44 | 0.9533 | 0.9467 | 0.0467 | 0.0533 |
| controle/debertinha-s42 | 0.7717 | 0.7483 | 0.2283 | 0.2517 |
| bertimbau-s44 | 0.9467 | 0.9417 | 0.0533 | 0.0583 |
| controle/bertimbau-s42 | 0.7900 | 0.7767 | 0.2100 | 0.2233 |
| albertina-s42 | 0.9617 | 0.9500 | 0.0383 | 0.0500 |
| controle/albertina-s42 | 0.8483 | 0.8267 | 0.1517 | 0.1733 |
| baseline_logreg | 0.9433 | 0.9300 | 0.0567 | 0.0700 |
| baseline_svm | 0.9417 | 0.9283 | 0.0583 | 0.0717 |

### Queda do acerto por faixa de WER (referência - transcrição, mesmos itens)

n por faixa: 0: 197, (0;0,1]: 76, (0,1;0,3]: 162, >0,3: 165. Queda = 1 se o modelo acertou na referência e errou na transcrição, -1 no contrário.

| modelo | 0 | (0;0,1] | (0,1;0,3] | >0,3 | queda média | Spearman ρ (WER × queda) | p |
|---|---|---|---|---|---|---|---|
| debertinha-s44 | 0.0000 | 0.0000 | 0.0247 | 0.0000 | 0.0067 | 0.0186 | 0.6501 |
| controle/debertinha-s42 | 0.0000 | 0.0000 | 0.0309 | 0.0545 | 0.0233 | 0.0704 | 0.0849 |
| bertimbau-s44 | 0.0000 | 0.0000 | 0.0000 | 0.0182 | 0.0050 | 0.0492 | 0.2288 |
| controle/bertimbau-s42 | 0.0000 | 0.0132 | 0.0062 | 0.0364 | 0.0133 | 0.0333 | 0.4162 |
| albertina-s42 | 0.0000 | 0.0000 | 0.0123 | 0.0303 | 0.0117 | 0.0787 | 0.0539 |
| controle/albertina-s42 | 0.0000 | 0.0000 | 0.0309 | 0.0485 | 0.0217 | 0.0658 | 0.1075 |
| baseline_logreg | 0.0000 | 0.0000 | 0.0309 | 0.0182 | 0.0133 | 0.0549 | 0.1796 |
| baseline_svm | 0.0000 | 0.0000 | 0.0309 | 0.0182 | 0.0133 | 0.0606 | 0.1384 |
