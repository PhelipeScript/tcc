# Sistema completo - publico (600 itens), baseline_svm

ativação = fração de itens em que o sistema decidiu DD (em DD é o acerto; em NDD e RUÍDO, falso positivo). 'política baixa→NDD' = como ficaria se confiabilidade baixa forçasse NDD.

| grupo | rótulo | n | ativação | ativação (política) | sem fala | falas/item | latência p50 (s) | latência p95 (s) |
|---|---|---|---|---|---|---|---|---|
| coraa | NDD | 300 | 0.1033 | 0.0867 | 0.0067 | 1.1567 | 1.1167 | 1.9474 |
| nurcsp | NDD | 300 | 0.0700 | 0.0633 | 0.0067 | 1.1467 | 1.1457 | 1.5561 |
