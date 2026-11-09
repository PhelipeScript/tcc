# Hipótese nas transcrições do Whisper - tts (740 itens)

Os mesmos itens para todos os modelos: o texto de referência (o que foi lido) e a transcrição do Whisper. Bootstrap pareado (2.000 reamostras) nas transcrições, como no teste interno: A - B, IC 95% e P(A melhor); no EER, menor é melhor.

### Na transcrição do Whisper

| modelo | acurácia | precisão DD | sensibilidade DD | F1 macro | EER |
|---|---|---|---|---|---|
| debertinha-s44 | 0.9743 | 0.9808 | 0.9676 | 0.9743 | 0.0216 |
| bertimbau-s44 | 0.9743 | 0.9808 | 0.9676 | 0.9743 | 0.0243 |
| albertina-s42 | 0.9824 | 0.9890 | 0.9757 | 0.9824 | 0.0216 |
| baseline_logreg | 0.9486 | 0.9637 | 0.9324 | 0.9486 | 0.0459 |
| baseline_svm | 0.9486 | 0.9536 | 0.9432 | 0.9486 | 0.0514 |

### No texto de referência

| modelo | acurácia | precisão DD | sensibilidade DD | F1 macro | EER |
|---|---|---|---|---|---|
| debertinha-s44 | 0.9757 | 0.9809 | 0.9703 | 0.9757 | 0.0189 |
| bertimbau-s44 | 0.9784 | 0.9863 | 0.9703 | 0.9784 | 0.0189 |
| albertina-s42 | 0.9838 | 0.9891 | 0.9784 | 0.9838 | 0.0162 |
| baseline_logreg | 0.9622 | 0.9698 | 0.9541 | 0.9622 | 0.0378 |
| baseline_svm | 0.9622 | 0.9572 | 0.9676 | 0.9622 | 0.0405 |

### Queda do acerto por faixa de WER (texto de referência - transcrição, mesmos itens)

n por faixa: 0: 580, (0;0,1]: 49, (0,1;0,3]: 78, >0,3: 33. Spearman entre o WER e a queda, item a item (queda = 1 se o modelo acertou no texto e errou na transcrição, -1 no contrário, 0 se não mudou).

| modelo | 0 | (0;0,1] | (0,1;0,3] | >0,3 | queda média | Spearman ρ | p |
|---|---|---|---|---|---|---|---|
| debertinha-s44 | 0.0000 | 0.0000 | -0.0128 | 0.0606 | 0.0014 | 0.0448 | 0.2240 |
| bertimbau-s44 | 0.0000 | 0.0000 | 0.0128 | 0.0606 | 0.0041 | 0.0885 | 0.0161 |
| albertina-s42 | 0.0000 | 0.0000 | -0.0128 | 0.0606 | 0.0014 | 0.0381 | 0.3011 |
| baseline_logreg | 0.0000 | 0.0000 | 0.0897 | 0.0909 | 0.0135 | 0.2000 | 0.0000 |
| baseline_svm | 0.0000 | 0.0000 | 0.1026 | 0.0606 | 0.0135 | 0.2177 | 0.0000 |

### Diferenças pareadas na transcrição

| comparação | métrica | diferença | IC95 inf | IC95 sup | P(A melhor) |
|---|---|---|---|---|---|
| debertinha-s44 vs bertimbau-s44 | accuracy | 0.0003 | -0.0095 | 0.0108 | 0.4770 |
| debertinha-s44 vs bertimbau-s44 | f1_macro | 0.0003 | -0.0095 | 0.0108 | 0.5225 |
| debertinha-s44 vs bertimbau-s44 | roc_auc | -0.0006 | -0.0029 | 0.0012 | 0.3060 |
| debertinha-s44 vs bertimbau-s44 | eer | -0.0013 | -0.0130 | 0.0104 | 0.5970 |
| debertinha-s44 vs albertina-s42 | accuracy | -0.0079 | -0.0176 | 0.0014 | 0.0295 |
| debertinha-s44 vs albertina-s42 | f1_macro | -0.0079 | -0.0176 | 0.0014 | 0.0380 |
| debertinha-s44 vs albertina-s42 | roc_auc | -0.0013 | -0.0037 | 0.0002 | 0.0685 |
| debertinha-s44 vs albertina-s42 | eer | 0.0023 | -0.0078 | 0.0133 | 0.3195 |
| debertinha-s44 vs baseline_logreg | accuracy | 0.0259 | 0.0108 | 0.0432 | 1.0000 |
| debertinha-s44 vs baseline_logreg | f1_macro | 0.0260 | 0.0108 | 0.0433 | 1.0000 |
| debertinha-s44 vs baseline_logreg | roc_auc | 0.0075 | 0.0029 | 0.0129 | 1.0000 |
| debertinha-s44 vs baseline_logreg | eer | -0.0255 | -0.0417 | -0.0105 | 1.0000 |
| debertinha-s44 vs baseline_svm | accuracy | 0.0258 | 0.0108 | 0.0419 | 0.9995 |
| debertinha-s44 vs baseline_svm | f1_macro | 0.0258 | 0.0108 | 0.0419 | 1.0000 |
| debertinha-s44 vs baseline_svm | roc_auc | 0.0083 | 0.0028 | 0.0150 | 0.9985 |
| debertinha-s44 vs baseline_svm | eer | -0.0284 | -0.0449 | -0.0132 | 1.0000 |
| bertimbau-s44 vs albertina-s42 | accuracy | -0.0083 | -0.0176 | 0.0014 | 0.0320 |
| bertimbau-s44 vs albertina-s42 | f1_macro | -0.0083 | -0.0176 | 0.0014 | 0.0400 |
| bertimbau-s44 vs albertina-s42 | roc_auc | -0.0007 | -0.0019 | 0.0002 | 0.0640 |
| bertimbau-s44 vs albertina-s42 | eer | 0.0037 | -0.0057 | 0.0132 | 0.2125 |
| bertimbau-s44 vs baseline_logreg | accuracy | 0.0256 | 0.0081 | 0.0432 | 0.9970 |
| bertimbau-s44 vs baseline_logreg | f1_macro | 0.0256 | 0.0082 | 0.0433 | 0.9970 |
| bertimbau-s44 vs baseline_logreg | roc_auc | 0.0081 | 0.0035 | 0.0138 | 1.0000 |
| bertimbau-s44 vs baseline_logreg | eer | -0.0241 | -0.0405 | -0.0082 | 0.9985 |
| bertimbau-s44 vs baseline_svm | accuracy | 0.0254 | 0.0081 | 0.0419 | 0.9970 |
| bertimbau-s44 vs baseline_svm | f1_macro | 0.0255 | 0.0081 | 0.0422 | 0.9970 |
| bertimbau-s44 vs baseline_svm | roc_auc | 0.0089 | 0.0033 | 0.0160 | 1.0000 |
| bertimbau-s44 vs baseline_svm | eer | -0.0270 | -0.0444 | -0.0104 | 0.9990 |
| albertina-s42 vs baseline_logreg | accuracy | 0.0339 | 0.0176 | 0.0514 | 1.0000 |
| albertina-s42 vs baseline_logreg | f1_macro | 0.0339 | 0.0176 | 0.0515 | 1.0000 |
| albertina-s42 vs baseline_logreg | roc_auc | 0.0088 | 0.0044 | 0.0143 | 1.0000 |
| albertina-s42 vs baseline_logreg | eer | -0.0278 | -0.0452 | -0.0114 | 0.9995 |
| albertina-s42 vs baseline_svm | accuracy | 0.0337 | 0.0176 | 0.0514 | 1.0000 |
| albertina-s42 vs baseline_svm | f1_macro | 0.0337 | 0.0176 | 0.0513 | 1.0000 |
| albertina-s42 vs baseline_svm | roc_auc | 0.0097 | 0.0042 | 0.0164 | 1.0000 |
| albertina-s42 vs baseline_svm | eer | -0.0307 | -0.0484 | -0.0138 | 1.0000 |
| baseline_logreg vs baseline_svm | accuracy | -0.0002 | -0.0081 | 0.0081 | 0.4080 |
| baseline_logreg vs baseline_svm | f1_macro | -0.0002 | -0.0081 | 0.0081 | 0.4545 |
| baseline_logreg vs baseline_svm | roc_auc | 0.0009 | -0.0010 | 0.0031 | 0.7805 |
| baseline_logreg vs baseline_svm | eer | -0.0029 | -0.0107 | 0.0054 | 0.8255 |
