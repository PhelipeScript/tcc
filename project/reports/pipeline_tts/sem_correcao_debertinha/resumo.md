# Sistema completo - tts, DeBERTinha sem correção (mesmas transcrições)

Este modelo julgou as transcrições da execução em tempo real com o BERTimbau (as mesmas das outras execuções); a latência ponta a ponta é a daquela execução.

ativação = fração de itens em que o sistema decidiu DD (em DD é o acerto; em NDD e RUÍDO, falso positivo). 'política baixa→NDD' = como ficaria se confiabilidade baixa forçasse NDD.

| grupo | rótulo | n | ativação | ativação (política) | sem fala | falas/item | latência p50 (s) | latência p95 (s) |
|---|---|---|---|---|---|---|---|---|
| tts:pt-BR-AntonioNeural | DD | 109 | 0.9541 | 0.9450 | 0.0000 | 1.0183 | None | None |
| tts:pt-BR-AntonioNeural | NDD | 138 | 0.3913 | 0.3913 | 0.0000 | 1.0652 | None | None |
| tts:pt-BR-FranciscaNeural | DD | 123 | 0.9431 | 0.9431 | 0.0000 | 1.0000 | None | None |
| tts:pt-BR-FranciscaNeural | NDD | 124 | 0.3871 | 0.3871 | 0.0000 | 1.0000 | None | None |
| tts:pt-BR-ThalitaMultilingualNeural | DD | 138 | 0.9275 | 0.9275 | 0.0000 | 1.0000 | None | None |
| tts:pt-BR-ThalitaMultilingualNeural | NDD | 108 | 0.2963 | 0.2963 | 0.0000 | 1.0000 | None | None |
