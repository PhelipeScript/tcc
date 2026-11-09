# Robustez ao ASR - publico: queda de acerto do texto de referência para a transcrição do Whisper

Queda = acerto no texto de referência - acerto na transcrição, nos mesmos itens. Diferença A - B com IC 95% por bootstrap pareado (2.000 reamostras); negativo = A perde menos.

| comparação | n | queda A | queda B | diferença | IC95 inf | IC95 sup | P(A perde menos) |
|---|---|---|---|---|---|---|---|
| bertimbau-s44 vs albertina-s42 | 600 | 0.0050 | 0.0117 | -0.0067 | -0.0217 | 0.0100 | 0.7700 |
| bertimbau-s44 vs debertinha-s44 | 600 | 0.0050 | 0.0067 | -0.0017 | -0.0200 | 0.0150 | 0.5385 |
| bertimbau-s44 vs baseline_logreg | 600 | 0.0050 | 0.0133 | -0.0083 | -0.0267 | 0.0100 | 0.7870 |
| bertimbau-s44 vs baseline_svm | 600 | 0.0050 | 0.0133 | -0.0083 | -0.0267 | 0.0100 | 0.7760 |
| albertina-s42 vs baseline_logreg | 600 | 0.0117 | 0.0133 | -0.0017 | -0.0217 | 0.0167 | 0.5335 |
| albertina-s42 vs baseline_svm | 600 | 0.0117 | 0.0133 | -0.0017 | -0.0217 | 0.0183 | 0.5395 |
| debertinha-s44 vs baseline_logreg | 600 | 0.0067 | 0.0133 | -0.0067 | -0.0267 | 0.0133 | 0.7185 |
| debertinha-s44 vs baseline_svm | 600 | 0.0067 | 0.0133 | -0.0067 | -0.0283 | 0.0133 | 0.7140 |
