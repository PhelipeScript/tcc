# Treino dos classificadores

Gerado por `scripts/relatorio_treino.py` a partir de `models/` (fora do git). O mesmo treino vale para as 3 sementes e para o modelo sem correção de cada família. Tempo, memória e precisão: `reports/custo/`; teste: `reports/experimento/`.

## Hiperparâmetros

| Parâmetro | BERTimbau | Albertina | DeBERTinha |
|---|---|---|---|
| Checkpoint base | `neuralmind/bert-base-portuguese-cased` | `PORTULAN/albertina-100m-portuguese-ptbr-encoder` | `sagui-nlp/debertinha-ptbr-xsmall` |
| Taxa de aprendizado | 2e-5 | 2e-5 | 2e-5 |
| Lote | 32 | 32 | 32 |
| Épocas (máximo) | 4 | 4 | 5 |
| Tokens por frase (máximo) | 128 | 128 | 128 |
| Aquecimento | 10% dos passos | 10% dos passos | 10% dos passos |
| Agendador | linear | linear | linear |
| Dropout | 0,1 | 0,1 | 0,1 |
| Weight decay | 0,01 | 0,01 | 0,01 |
| Parada antecipada | paciência 2, f1_macro | paciência 2, f1_macro | paciência 2, f1_macro |
| Precisão | bf16 | fp32 | bf16 |

## Por execução

| Execução | Corpus | Melhor checkpoint | F1 macro de validação (limiar 0,5) | Limiar escolhido | No sistema |
|---|---|---|---|---|---|
| bertimbau-s42 | com correção | época 4 de 4 | 0,9830 | 0,420 |  |
| bertimbau-s43 | com correção | época 4 de 4 | 0,9833 | 0,725 |  |
| bertimbau-s44 | com correção | época 3 de 4 | 0,9834 | 0,685 | sim |
| albertina-s42 | com correção | época 4 de 4 | 0,9792 | 0,620 | sim |
| albertina-s43 | com correção | época 4 de 4 | 0,9787 | 0,940 |  |
| albertina-s44 | com correção | época 4 de 4 | 0,9789 | 0,555 |  |
| debertinha-s42 | com correção | época 5 de 5 | 0,9787 | 0,830 |  |
| debertinha-s43 | com correção | época 5 de 5 | 0,9793 | 0,800 |  |
| debertinha-s44 | com correção | época 5 de 5 | 0,9798 | 0,395 | sim |
| controle/bertimbau-s42 | sem correção | época 4 de 4 | 0,9887 | 0,070 |  |
| controle/albertina-s42 | sem correção | época 4 de 4 | 0,9879 | 0,490 |  |
| controle/debertinha-s42 | sem correção | época 5 de 5 | 0,9874 | 0,030 |  |

## F1 macro de validação por época, durante o treino

| Execução | Época 1 | Época 2 | Época 3 | Época 4 | Época 5 |
|---|---|---|---|---|---|
| bertimbau-s42 | 0,9747 | 0,9800 | 0,9824 | 0,9831 |  |
| bertimbau-s43 | 0,9772 | 0,9807 | 0,9813 | 0,9833 |  |
| bertimbau-s44 | 0,9758 | 0,9807 | 0,9834 | 0,9833 |  |
| albertina-s42 | 0,9694 | 0,9758 | 0,9756 | 0,9792 |  |
| albertina-s43 | 0,9703 | 0,9748 | 0,9783 | 0,9787 |  |
| albertina-s44 | 0,9704 | 0,9749 | 0,9777 | 0,9789 |  |
| debertinha-s42 | 0,9686 | 0,9750 | 0,9776 | 0,9769 | 0,9788 |
| debertinha-s43 | 0,9688 | 0,9752 | 0,9751 | 0,9782 | 0,9790 |
| debertinha-s44 | 0,9695 | 0,9752 | 0,9749 | 0,9781 | 0,9797 |
| controle/bertimbau-s42 | 0,9853 | 0,9879 | 0,9886 | 0,9887 |  |
| controle/albertina-s42 | 0,9823 | 0,9844 | 0,9859 | 0,9879 |  |
| controle/debertinha-s42 | 0,9806 | 0,9860 | 0,9871 | 0,9866 | 0,9874 |

## Perda de validação por época, durante o treino

| Execução | Época 1 | Época 2 | Época 3 | Época 4 | Época 5 |
|---|---|---|---|---|---|
| bertimbau-s42 | 0,0783 | 0,0741 | 0,0727 | 0,0869 |  |
| bertimbau-s43 | 0,0760 | 0,0647 | 0,0866 | 0,0888 |  |
| bertimbau-s44 | 0,0783 | 0,0789 | 0,0750 | 0,0819 |  |
| albertina-s42 | 0,0977 | 0,0823 | 0,0972 | 0,1068 |  |
| albertina-s43 | 0,0930 | 0,0868 | 0,0941 | 0,1035 |  |
| albertina-s44 | 0,1018 | 0,0895 | 0,0915 | 0,1108 |  |
| debertinha-s42 | 0,1032 | 0,0851 | 0,0842 | 0,0995 | 0,0910 |
| debertinha-s43 | 0,0976 | 0,0860 | 0,0887 | 0,0815 | 0,0843 |
| debertinha-s44 | 0,1025 | 0,0978 | 0,0838 | 0,0881 | 0,0863 |
| controle/bertimbau-s42 | 0,0492 | 0,0449 | 0,0519 | 0,0590 |  |
| controle/albertina-s42 | 0,0528 | 0,0577 | 0,0590 | 0,0590 |  |
| controle/debertinha-s42 | 0,0629 | 0,0482 | 0,0493 | 0,0544 | 0,0531 |

## TF-IDF

| Modelo | C (validação cruzada de 5 partes) | F1 macro na validação cruzada | F1 macro de validação | Limiar escolhido |
|---|---|---|---|---|
| TF-IDF + Regressão Logística | 10,0 | 0,9585 | 0,9605 | 0,480 |
| TF-IDF + SVM | 1,0 | 0,9608 | 0,9611 | 0,460 |

Notas:
- Em 11 de 12 execuções, a menor perda de validação vem numa época anterior à do melhor checkpoint: depois dela, a perda de validação sobe, mas a F1 de validação ainda melhora. O checkpoint é escolhido pela F1.
- A validação dos modelos sem correção é a do corpus deles, que ainda tem o atalho do tamanho.
- O melhor checkpoint é o de maior F1 macro de validação; o limiar maximiza a mesma F1 na validação.
- "No sistema": a semente de maior F1 macro de validação (limiar 0,5), usada no sistema em tempo real.
