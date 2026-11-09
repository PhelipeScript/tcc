# Robustez ao ASR - gravacoes: queda de acerto do texto de referência para a transcrição do Whisper

Queda = acerto no texto de referência - acerto na transcrição, nos mesmos itens. Diferença A - B com IC 95% por bootstrap pareado (2.000 reamostras); negativo = A perde menos. O bootstrap sorteia as frases: as leituras da mesma frase nos 4 cenários entram juntas.

| comparação | n | queda A | queda B | diferença | IC95 inf | IC95 sup | P(A perde menos) |
|---|---|---|---|---|---|---|---|
| bertimbau-s44 vs albertina-s42 | 184 | 0.0217 | 0.0163 | 0.0054 | -0.0361 | 0.0612 | 0.4060 |
| bertimbau-s44 vs debertinha-s44 | 184 | 0.0217 | 0.0217 | 0.0000 | -0.0446 | 0.0579 | 0.4925 |
| bertimbau-s44 vs baseline_logreg | 184 | 0.0217 | 0.0489 | -0.0272 | -0.0857 | 0.0408 | 0.7895 |
| bertimbau-s44 vs baseline_svm | 184 | 0.0217 | 0.0489 | -0.0272 | -0.0857 | 0.0408 | 0.7895 |
| albertina-s42 vs baseline_logreg | 184 | 0.0163 | 0.0489 | -0.0326 | -0.0843 | 0.0120 | 0.9045 |
| albertina-s42 vs baseline_svm | 184 | 0.0163 | 0.0489 | -0.0326 | -0.0843 | 0.0120 | 0.9045 |
| debertinha-s44 vs baseline_logreg | 184 | 0.0217 | 0.0489 | -0.0272 | -0.0792 | 0.0169 | 0.8525 |
| debertinha-s44 vs baseline_svm | 184 | 0.0217 | 0.0489 | -0.0272 | -0.0792 | 0.0169 | 0.8525 |
