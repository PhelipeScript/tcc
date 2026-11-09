# Sistema completo - tts (740 itens), baseline_svm

ativação = fração de itens em que o sistema decidiu DD (em DD é o acerto; em NDD e RUÍDO, falso positivo). 'política baixa→NDD' = como ficaria se confiabilidade baixa forçasse NDD.

| grupo | rótulo | n | ativação | ativação (política) | sem fala | falas/item | latência p50 (s) | latência p95 (s) |
|---|---|---|---|---|---|---|---|---|
| tts:pt-BR-AntonioNeural | DD | 109 | 0.9541 | 0.9450 | 0.0000 | 1.0183 | 1.1202 | 1.1908 |
| tts:pt-BR-AntonioNeural | NDD | 138 | 0.0362 | 0.0362 | 0.0000 | 1.0652 | 1.1202 | 1.1908 |
| tts:pt-BR-FranciscaNeural | DD | 123 | 0.9268 | 0.9268 | 0.0000 | 1.0000 | 1.1233 | 1.2010 |
| tts:pt-BR-FranciscaNeural | NDD | 124 | 0.0484 | 0.0484 | 0.0000 | 1.0000 | 1.1233 | 1.2010 |
| tts:pt-BR-ThalitaMultilingualNeural | DD | 138 | 0.9493 | 0.9493 | 0.0000 | 1.0000 | 1.1287 | 1.2057 |
| tts:pt-BR-ThalitaMultilingualNeural | NDD | 108 | 0.0741 | 0.0741 | 0.0000 | 1.0000 | 1.1287 | 1.2057 |
