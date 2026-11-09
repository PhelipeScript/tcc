# Hipótese nas transcrições do Whisper - gravacoes (184 itens)

Os mesmos itens para todos os modelos: o texto de referência (o que foi lido) e a transcrição do Whisper. Bootstrap pareado (2.000 reamostras) nas transcrições, como no teste interno: A - B, IC 95% e P(A melhor); no EER, menor é melhor. O bootstrap sorteia as frases: as leituras da mesma frase nos 4 cenários entram juntas.

### Na transcrição do Whisper

| modelo | acurácia | precisão DD | sensibilidade DD | F1 macro | EER |
|---|---|---|---|---|---|
| debertinha-s44 | 0.9348 | 0.9359 | 0.9125 | 0.9335 | 0.0625 |
| bertimbau-s44 | 0.9348 | 0.9146 | 0.9375 | 0.9338 | 0.0625 |
| albertina-s42 | 0.9402 | 0.9481 | 0.9125 | 0.9389 | 0.0673 |
| baseline_logreg | 0.9239 | 0.9583 | 0.8625 | 0.9215 | 0.1250 |
| baseline_svm | 0.9022 | 0.9079 | 0.8625 | 0.8999 | 0.0875 |

### No texto de referência

| modelo | acurácia | precisão DD | sensibilidade DD | F1 macro | EER |
|---|---|---|---|---|---|
| debertinha-s44 | 0.9565 | 0.9500 | 0.9500 | 0.9558 | 0.0385 |
| bertimbau-s44 | 0.9565 | 0.9500 | 0.9500 | 0.9558 | 0.0385 |
| albertina-s42 | 0.9565 | 0.9500 | 0.9500 | 0.9558 | 0.0500 |
| baseline_logreg | 0.9728 | 0.9870 | 0.9500 | 0.9722 | 0.0500 |
| baseline_svm | 0.9511 | 0.9383 | 0.9500 | 0.9503 | 0.0481 |

### Queda do acerto por faixa de WER (texto de referência - transcrição, mesmos itens)

n por faixa: 0: 86, (0;0,1]: 14, (0,1;0,3]: 48, >0,3: 36. Spearman entre o WER e a queda, item a item (queda = 1 se o modelo acertou no texto e errou na transcrição, -1 no contrário, 0 se não mudou). O p trata as leituras da mesma frase como independentes, então é otimista.

| modelo | 0 | (0;0,1] | (0,1;0,3] | >0,3 | queda média | Spearman ρ | p |
|---|---|---|---|---|---|---|---|
| debertinha-s44 | 0.0000 | 0.0000 | 0.0000 | 0.1111 | 0.0217 | 0.1832 | 0.0128 |
| bertimbau-s44 | 0.0000 | 0.0714 | 0.0417 | 0.0278 | 0.0217 | 0.0603 | 0.4159 |
| albertina-s42 | 0.0000 | 0.0000 | 0.0000 | 0.0833 | 0.0163 | 0.1452 | 0.0493 |
| baseline_logreg | 0.0000 | 0.0714 | 0.0417 | 0.1667 | 0.0489 | 0.2730 | 0.0002 |
| baseline_svm | 0.0000 | 0.0714 | 0.0417 | 0.1667 | 0.0489 | 0.2730 | 0.0002 |

### Diferenças pareadas na transcrição

| comparação | métrica | diferença | IC95 inf | IC95 sup | P(A melhor) |
|---|---|---|---|---|---|
| debertinha-s44 vs bertimbau-s44 | accuracy | -0.0001 | -0.0357 | 0.0442 | 0.4195 |
| debertinha-s44 vs bertimbau-s44 | f1_macro | -0.0004 | -0.0369 | 0.0449 | 0.4300 |
| debertinha-s44 vs bertimbau-s44 | roc_auc | -0.0085 | -0.0304 | 0.0045 | 0.1640 |
| debertinha-s44 vs bertimbau-s44 | eer | 0.0034 | -0.0421 | 0.0437 | 0.3685 |
| debertinha-s44 vs albertina-s42 | accuracy | -0.0054 | -0.0169 | 0.0000 | 0.0000 |
| debertinha-s44 vs albertina-s42 | f1_macro | -0.0056 | -0.0173 | 0.0000 | 0.0000 |
| debertinha-s44 vs albertina-s42 | roc_auc | 0.0040 | -0.0193 | 0.0274 | 0.6460 |
| debertinha-s44 vs albertina-s42 | eer | -0.0038 | -0.0395 | 0.0294 | 0.5845 |
| debertinha-s44 vs baseline_logreg | accuracy | 0.0099 | -0.0588 | 0.0736 | 0.6050 |
| debertinha-s44 vs baseline_logreg | f1_macro | 0.0113 | -0.0607 | 0.0791 | 0.6450 |
| debertinha-s44 vs baseline_logreg | roc_auc | 0.0315 | -0.0100 | 0.0808 | 0.9395 |
| debertinha-s44 vs baseline_logreg | eer | -0.0489 | -0.1429 | 0.0469 | 0.8505 |
| debertinha-s44 vs baseline_svm | accuracy | 0.0309 | -0.0457 | 0.1065 | 0.7770 |
| debertinha-s44 vs baseline_svm | f1_macro | 0.0327 | -0.0459 | 0.1128 | 0.7970 |
| debertinha-s44 vs baseline_svm | roc_auc | 0.0280 | -0.0151 | 0.0796 | 0.9080 |
| debertinha-s44 vs baseline_svm | eer | -0.0335 | -0.1296 | 0.0619 | 0.7530 |
| bertimbau-s44 vs albertina-s42 | accuracy | -0.0054 | -0.0481 | 0.0311 | 0.3390 |
| bertimbau-s44 vs albertina-s42 | f1_macro | -0.0051 | -0.0498 | 0.0317 | 0.3480 |
| bertimbau-s44 vs albertina-s42 | roc_auc | 0.0125 | 0.0000 | 0.0303 | 0.9780 |
| bertimbau-s44 vs albertina-s42 | eer | -0.0072 | -0.0377 | 0.0112 | 0.6875 |
| bertimbau-s44 vs baseline_logreg | accuracy | 0.0099 | -0.0608 | 0.0743 | 0.5895 |
| bertimbau-s44 vs baseline_logreg | f1_macro | 0.0118 | -0.0603 | 0.0800 | 0.6465 |
| bertimbau-s44 vs baseline_logreg | roc_auc | 0.0400 | 0.0071 | 0.0869 | 0.9970 |
| bertimbau-s44 vs baseline_logreg | eer | -0.0522 | -0.1409 | 0.0283 | 0.8950 |
| bertimbau-s44 vs baseline_svm | accuracy | 0.0310 | -0.0467 | 0.1070 | 0.7615 |
| bertimbau-s44 vs baseline_svm | f1_macro | 0.0332 | -0.0475 | 0.1121 | 0.7965 |
| bertimbau-s44 vs baseline_svm | roc_auc | 0.0365 | 0.0024 | 0.0849 | 0.9855 |
| bertimbau-s44 vs baseline_svm | eer | -0.0368 | -0.1286 | 0.0459 | 0.7945 |
| albertina-s42 vs baseline_logreg | accuracy | 0.0153 | -0.0535 | 0.0798 | 0.6595 |
| albertina-s42 vs baseline_logreg | f1_macro | 0.0169 | -0.0565 | 0.0863 | 0.7005 |
| albertina-s42 vs baseline_logreg | roc_auc | 0.0275 | -0.0019 | 0.0671 | 0.9665 |
| albertina-s42 vs baseline_logreg | eer | -0.0451 | -0.1398 | 0.0410 | 0.8460 |
| albertina-s42 vs baseline_svm | accuracy | 0.0363 | -0.0390 | 0.1124 | 0.8060 |
| albertina-s42 vs baseline_svm | f1_macro | 0.0383 | -0.0407 | 0.1203 | 0.8300 |
| albertina-s42 vs baseline_svm | roc_auc | 0.0240 | -0.0059 | 0.0669 | 0.9335 |
| albertina-s42 vs baseline_svm | eer | -0.0297 | -0.1184 | 0.0589 | 0.7395 |
| baseline_logreg vs baseline_svm | accuracy | 0.0211 | 0.0000 | 0.0674 | 0.6275 |
| baseline_logreg vs baseline_svm | f1_macro | 0.0214 | 0.0000 | 0.0696 | 0.6275 |
| baseline_logreg vs baseline_svm | roc_auc | -0.0035 | -0.0209 | 0.0065 | 0.3635 |
| baseline_logreg vs baseline_svm | eer | 0.0154 | -0.0326 | 0.0952 | 0.3275 |
