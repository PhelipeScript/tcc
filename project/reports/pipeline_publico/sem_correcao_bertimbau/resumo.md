# Sistema completo - publico, BERTimbau sem correção (mesmas transcrições)

Este modelo julgou as transcrições da execução em tempo real com o BERTimbau (as mesmas das outras execuções); a latência ponta a ponta é a daquela execução.

ativação = fração de itens em que o sistema decidiu DD (em DD é o acerto; em NDD e RUÍDO, falso positivo). 'política baixa→NDD' = como ficaria se confiabilidade baixa forçasse NDD.

| grupo | rótulo | n | ativação | ativação (política) | sem fala | falas/item | latência p50 (s) | latência p95 (s) |
|---|---|---|---|---|---|---|---|---|
| coraa | NDD | 300 | 0.3467 | 0.2767 | 0.0067 | 1.1567 | None | None |
| nurcsp | NDD | 300 | 0.2367 | 0.1800 | 0.0067 | 1.1467 | None | None |
