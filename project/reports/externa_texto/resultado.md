# Validação externa em texto - taxa de falso positivo (NDD por suposição de domínio)

2912 frases de conversa real (coraa, nurcsp, tagarela), sem as iguais ou quase iguais ao corpus. Toda ativação conta como falso positivo.

| execução | coraa | nurcsp | tagarela | geral |
|---|---|---|---|---|
| debertinha-s42 | 0.0340 | 0.0390 | 0.0581 | 0.0433 |
| debertinha-s43 | 0.0440 | 0.0340 | 0.0450 | 0.0409 |
| debertinha-s44 | 0.0510 | 0.0590 | 0.0603 | 0.0567 |
| bertimbau-s42 | 0.0400 | 0.0490 | 0.0625 | 0.0501 |
| bertimbau-s43 | 0.0410 | 0.0560 | 0.0768 | 0.0573 |
| bertimbau-s44 | 0.0470 | 0.0660 | 0.0779 | 0.0632 |
| albertina-s42 | 0.0350 | 0.0530 | 0.0746 | 0.0536 |
| albertina-s43 | 0.0410 | 0.0410 | 0.0669 | 0.0491 |
| albertina-s44 | 0.0370 | 0.0650 | 0.0866 | 0.0622 |
| controle/debertinha-s42 | 0.4140 | 0.2390 | 0.2675 | 0.3080 |
| controle/bertimbau-s42 | 0.3490 | 0.2050 | 0.2610 | 0.2720 |
| controle/albertina-s42 | 0.2810 | 0.1620 | 0.2171 | 0.2201 |
| baseline_logreg | 0.0630 | 0.0630 | 0.0559 | 0.0608 |
| baseline_svm | 0.0680 | 0.0660 | 0.0636 | 0.0659 |

### FP geral por semente (condição principal)

A semente usada no pipeline é a de maior F1 macro de validação; o teste e os dados externos nunca orientam essa escolha.

| modelo | s42 | s43 | s44 | média | desvio | escolhida pela validação |
|---|---|---|---|---|---|---|
| debertinha | 0.0433 | 0.0409 | 0.0567 | 0.0469 | 0.0085 | debertinha-s44 |
| bertimbau | 0.0501 | 0.0573 | 0.0632 | 0.0569 | 0.0065 | bertimbau-s44 |
| albertina | 0.0536 | 0.0491 | 0.0622 | 0.0549 | 0.0066 | albertina-s42 |

### Bootstrap pareado da diferença de FP (A - B; negativo = A ativa menos)

| comparação | FP A | FP B | diferença | IC95 inf | IC95 sup | P(A menos FP) |
|---|---|---|---|---|---|---|
| debertinha-s44 vs baseline_logreg | 0.0567 | 0.0608 | -0.0041 | -0.0148 | 0.0065 | 0.7545 |
| debertinha-s44 vs baseline_svm | 0.0567 | 0.0659 | -0.0093 | -0.0203 | 0.0017 | 0.9430 |
| debertinha-s44 vs controle/debertinha-s42 | 0.0567 | 0.3080 | -0.2514 | -0.2692 | -0.2342 | 1.0000 |
| bertimbau-s44 vs baseline_logreg | 0.0632 | 0.0608 | 0.0024 | -0.0082 | 0.0137 | 0.3080 |
| bertimbau-s44 vs baseline_svm | 0.0632 | 0.0659 | -0.0027 | -0.0134 | 0.0086 | 0.6505 |
| bertimbau-s44 vs controle/bertimbau-s42 | 0.0632 | 0.2720 | -0.2088 | -0.2256 | -0.1930 | 1.0000 |
| albertina-s42 vs baseline_logreg | 0.0536 | 0.0608 | -0.0072 | -0.0175 | 0.0031 | 0.8965 |
| albertina-s42 vs baseline_svm | 0.0536 | 0.0659 | -0.0124 | -0.0227 | -0.0010 | 0.9895 |
| albertina-s42 vs controle/albertina-s42 | 0.0536 | 0.2201 | -0.1666 | -0.1823 | -0.1518 | 1.0000 |
| bertimbau-s44 vs albertina-s42 | 0.0632 | 0.0536 | 0.0096 | 0.0014 | 0.0182 | 0.0110 |
| bertimbau-s44 vs debertinha-s44 | 0.0632 | 0.0567 | 0.0065 | -0.0017 | 0.0148 | 0.0580 |
| albertina-s42 vs debertinha-s44 | 0.0536 | 0.0567 | -0.0031 | -0.0120 | 0.0058 | 0.7330 |
