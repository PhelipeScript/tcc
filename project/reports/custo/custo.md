# Custo dos classificadores

Parâmetros contados nos pesos do melhor checkpoint; treino = média ± desvio das sementes (Mac M5, MPS); inferência por fala na CPU com as threads do pipeline, medida no sistema ao vivo na voz do autor (cada modelo na sua execução).

| modelo | base | parâmetros (mi) | tamanho (MB) | precisão do treino | treino (min) | pico de memória no treino (GB) | inferência p50 (ms) |
|---|---|---|---|---|---|---|---|
| bertimbau | neuralmind/bert-base-portuguese-cased | 108,9 | 416 | bf16 | 56 ± 5 | 3,4 | 41,1 |
| albertina | PORTULAN/albertina-100m-portuguese-ptbr-encoder | 139,2 | 531 | fp32 | 148 ± 3 | 8,8 | 43,5 |
| debertinha | sagui-nlp/debertinha-ptbr-xsmall | 40,9 | 156 | bf16 | 50 ± 3 | 2,4 | 31,1 |
| baseline_logreg | TF-IDF de 1 a 3-gramas, 50.000 termos | - | 0,9 | - | 0,08 | - | 1,11 |
| baseline_svm | TF-IDF de 1 a 3-gramas, 50.000 termos | - | 2,3 | - | 0,07 | - | 4,17 |

### Benchmark dos Transformers (benchmark-20261104-0737.json), p50 em ms por frase

| modelo | CPU 1 thread | CPU 2 threads | CPU 4 threads | MPS |
|---|---|---|---|---|
| bertimbau | 11,6 | 11,9 | 12,1 | 6,8 |
| debertinha | 8,3 | 9,0 | 9,2 | 9,6 |
| albertina | 15,4 | 15,9 | 15,9 | 10,9 |
