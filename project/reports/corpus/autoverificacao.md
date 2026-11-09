# Autoverificação do rótulo na geração

O mesmo modelo que gerou o lote o reclassifica em temperatura 0, sem ver o rótulo pretendido; só ficam as frases em que ele confirma o rótulo. No corpus final, 17904 das 116438 frases passaram por ela; o corpus inicial e as 3144 frases do Claude, não. Os logs cobrem 11823 das 18987 frases verificadas nas fontes; a taxa é parcial.

| escopo | nome | classe | julgadas | discordâncias | taxa de discordância | descartadas (resposta inválida) |
|---|---|---|---|---|---|---|
| total | total | DD | 6.076 | 61 | 0.0100 | 0 |
| total | total | NDD | 7.154 | 929 | 0.1299 | 177 |
| estratégia | comprimento | DD | 4.046 | 30 | 0.0074 | 0 |
| estratégia | comprimento | NDD | 4.905 | 848 | 0.1729 | 116 |
| estratégia | negativos_dificeis | DD | 2.030 | 31 | 0.0153 | 0 |
| estratégia | negativos_dificeis | NDD | 2.249 | 81 | 0.0360 | 61 |
| log | gemini35flash | DD | 69 | 0 | 0.0000 | 0 |
| log | gemini35flash | NDD | 80 | 22 | 0.2750 | 0 |
| log | gptoss120b | DD | 757 | 0 | 0.0000 | 0 |
| log | gptoss120b | NDD | 1.092 | 294 | 0.2692 | 20 |
| log | gptoss120b_negdif | DD | 922 | 0 | 0.0000 | 0 |
| log | gptoss120b_negdif | NDD | 990 | 44 | 0.0444 | 14 |
| log | lmstudio_500 | DD | 1.818 | 25 | 0.0138 | 0 |
| log | lmstudio_500 | NDD | 2.139 | 345 | 0.1613 | 0 |
| log | lmstudio_piloto | DD | 211 | 0 | 0.0000 | 0 |
| log | lmstudio_piloto | NDD | 142 | 38 | 0.2676 | 0 |
| log | lmstudio_v2_teste | DD | 78 | 0 | 0.0000 | 0 |
| log | lmstudio_v2_teste | NDD | 96 | 9 | 0.0938 | 0 |
| log | qwen38 | DD | 1.113 | 5 | 0.0045 | 0 |
| log | qwen38 | NDD | 1.356 | 140 | 0.1032 | 96 |
| log | qwen38_negdif | DD | 1.108 | 31 | 0.0280 | 0 |
| log | qwen38_negdif | NDD | 1.259 | 37 | 0.0294 | 47 |
