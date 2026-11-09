# Bootstrap pareado (2000 reamostragens, teste principal)

A - B; IC 95% por percentis; P(A melhor) = fração das reamostragens em que A supera B (no EER, menor é melhor). Semente de cada Transformer escolhida pela validação.

| comparação | métrica | diferença | IC95 inf | IC95 sup | P(A melhor) |
|---|---|---|---|---|---|
| debertinha vs bertimbau | accuracy | -0.0062 | -0.0086 | -0.0040 | 0.0000 |
| debertinha vs bertimbau | f1_macro | -0.0062 | -0.0086 | -0.0040 | 0.0000 |
| debertinha vs bertimbau | roc_auc | -0.0006 | -0.0011 | -0.0002 | 0.0070 |
| debertinha vs bertimbau | eer | 0.0050 | 0.0026 | 0.0074 | 0.0000 |
| debertinha vs albertina | accuracy | -0.0030 | -0.0053 | -0.0006 | 0.0050 |
| debertinha vs albertina | f1_macro | -0.0030 | -0.0053 | -0.0006 | 0.0050 |
| debertinha vs albertina | roc_auc | 0.0002 | -0.0003 | 0.0008 | 0.7710 |
| debertinha vs albertina | eer | 0.0012 | -0.0013 | 0.0038 | 0.1890 |
| debertinha vs baseline_logreg | accuracy | 0.0161 | 0.0127 | 0.0195 | 1.0000 |
| debertinha vs baseline_logreg | f1_macro | 0.0161 | 0.0127 | 0.0195 | 1.0000 |
| debertinha vs baseline_logreg | roc_auc | 0.0060 | 0.0049 | 0.0072 | 1.0000 |
| debertinha vs baseline_logreg | eer | -0.0173 | -0.0211 | -0.0137 | 1.0000 |
| debertinha vs baseline_svm | accuracy | 0.0155 | 0.0123 | 0.0188 | 1.0000 |
| debertinha vs baseline_svm | f1_macro | 0.0155 | 0.0123 | 0.0188 | 1.0000 |
| debertinha vs baseline_svm | roc_auc | 0.0066 | 0.0054 | 0.0080 | 1.0000 |
| debertinha vs baseline_svm | eer | -0.0179 | -0.0215 | -0.0144 | 1.0000 |
| bertimbau vs albertina | accuracy | 0.0032 | 0.0010 | 0.0054 | 0.9970 |
| bertimbau vs albertina | f1_macro | 0.0032 | 0.0010 | 0.0054 | 0.9975 |
| bertimbau vs albertina | roc_auc | 0.0008 | 0.0003 | 0.0014 | 0.9995 |
| bertimbau vs albertina | eer | -0.0038 | -0.0061 | -0.0015 | 1.0000 |
| bertimbau vs baseline_logreg | accuracy | 0.0223 | 0.0190 | 0.0257 | 1.0000 |
| bertimbau vs baseline_logreg | f1_macro | 0.0223 | 0.0190 | 0.0257 | 1.0000 |
| bertimbau vs baseline_logreg | roc_auc | 0.0066 | 0.0054 | 0.0079 | 1.0000 |
| bertimbau vs baseline_logreg | eer | -0.0224 | -0.0260 | -0.0188 | 1.0000 |
| bertimbau vs baseline_svm | accuracy | 0.0217 | 0.0185 | 0.0251 | 1.0000 |
| bertimbau vs baseline_svm | f1_macro | 0.0217 | 0.0185 | 0.0251 | 1.0000 |
| bertimbau vs baseline_svm | roc_auc | 0.0072 | 0.0059 | 0.0086 | 1.0000 |
| bertimbau vs baseline_svm | eer | -0.0229 | -0.0265 | -0.0194 | 1.0000 |
| albertina vs baseline_logreg | accuracy | 0.0191 | 0.0156 | 0.0226 | 1.0000 |
| albertina vs baseline_logreg | f1_macro | 0.0191 | 0.0156 | 0.0226 | 1.0000 |
| albertina vs baseline_logreg | roc_auc | 0.0058 | 0.0046 | 0.0070 | 1.0000 |
| albertina vs baseline_logreg | eer | -0.0185 | -0.0222 | -0.0149 | 1.0000 |
| albertina vs baseline_svm | accuracy | 0.0185 | 0.0152 | 0.0219 | 1.0000 |
| albertina vs baseline_svm | f1_macro | 0.0185 | 0.0152 | 0.0219 | 1.0000 |
| albertina vs baseline_svm | roc_auc | 0.0064 | 0.0051 | 0.0078 | 1.0000 |
| albertina vs baseline_svm | eer | -0.0191 | -0.0228 | -0.0157 | 1.0000 |
| baseline_logreg vs baseline_svm | accuracy | -0.0006 | -0.0021 | 0.0009 | 0.2165 |
| baseline_logreg vs baseline_svm | f1_macro | -0.0006 | -0.0021 | 0.0009 | 0.2380 |
| baseline_logreg vs baseline_svm | roc_auc | 0.0006 | 0.0003 | 0.0010 | 0.9990 |
| baseline_logreg vs baseline_svm | eer | -0.0005 | -0.0022 | 0.0012 | 0.7455 |
