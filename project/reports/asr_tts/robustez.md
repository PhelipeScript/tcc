# Robustez ao ASR - tts: queda de acerto do texto de referência para a transcrição do Whisper

Queda = acerto no texto de referência - acerto na transcrição, nos mesmos itens. Diferença A - B com IC 95% por bootstrap pareado (2.000 reamostras); negativo = A perde menos.

| comparação | n | queda A | queda B | diferença | IC95 inf | IC95 sup | P(A perde menos) |
|---|---|---|---|---|---|---|---|
| bertimbau-s44 vs albertina-s42 | 740 | 0.0041 | 0.0014 | 0.0027 | -0.0054 | 0.0108 | 0.1840 |
| bertimbau-s44 vs debertinha-s44 | 740 | 0.0041 | 0.0014 | 0.0027 | -0.0041 | 0.0095 | 0.1510 |
| bertimbau-s44 vs baseline_logreg | 740 | 0.0041 | 0.0135 | -0.0095 | -0.0203 | 0.0000 | 0.9550 |
| bertimbau-s44 vs baseline_svm | 740 | 0.0041 | 0.0135 | -0.0095 | -0.0203 | 0.0014 | 0.9565 |
| albertina-s42 vs baseline_logreg | 740 | 0.0014 | 0.0135 | -0.0122 | -0.0216 | -0.0041 | 0.9985 |
| albertina-s42 vs baseline_svm | 740 | 0.0014 | 0.0135 | -0.0122 | -0.0216 | -0.0027 | 0.9965 |
| debertinha-s44 vs baseline_logreg | 740 | 0.0014 | 0.0135 | -0.0122 | -0.0230 | -0.0027 | 0.9945 |
| debertinha-s44 vs baseline_svm | 740 | 0.0014 | 0.0135 | -0.0122 | -0.0216 | -0.0027 | 0.9865 |
