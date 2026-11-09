# Sistema completo - tts (740 itens), baseline_logreg

ativação = fração de itens em que o sistema decidiu DD (em DD é o acerto; em NDD e RUÍDO, falso positivo). 'política baixa→NDD' = como ficaria se confiabilidade baixa forçasse NDD.

| grupo | rótulo | n | ativação | ativação (política) | sem fala | falas/item | latência p50 (s) | latência p95 (s) |
|---|---|---|---|---|---|---|---|---|
| tts:pt-BR-AntonioNeural | DD | 109 | 0.9358 | 0.9266 | 0.0000 | 1.0183 | 1.1214 | 1.1913 |
| tts:pt-BR-AntonioNeural | NDD | 138 | 0.0217 | 0.0217 | 0.0000 | 1.0652 | 1.1214 | 1.1913 |
| tts:pt-BR-FranciscaNeural | DD | 123 | 0.9268 | 0.9268 | 0.0000 | 1.0000 | 1.1266 | 1.2037 |
| tts:pt-BR-FranciscaNeural | NDD | 124 | 0.0484 | 0.0484 | 0.0000 | 1.0000 | 1.1266 | 1.2037 |
| tts:pt-BR-ThalitaMultilingualNeural | DD | 138 | 0.9348 | 0.9348 | 0.0000 | 1.0000 | 1.1304 | 1.2000 |
| tts:pt-BR-ThalitaMultilingualNeural | NDD | 108 | 0.0556 | 0.0556 | 0.0000 | 1.0000 | 1.1304 | 1.2000 |
