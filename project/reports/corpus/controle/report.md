# Relatório de construção do corpus DDSD

Gerado em 2026-11-01T22:04:14-0300 em 11.7 s. Configuração: `config/corpus_controle.toml`.

## Funil

| etapa | DD | NDD | total | nota |
|---|---|---|---|---|
| fontes | 136.153 | 131.213 | 267.366 | descartadas na leitura: {'rotulo_divergente': 0, 'texto_ausente': 0} |
| normalização e filtros | 136.077 | 131.130 | 267.207 | descartes: {'tamanho': 39, 'nao_ascii': 50, 'lista_serializada': 70} |
| dedup exato + conflitos | 121.654 | 130.750 | 252.404 | duplicatas: 14717; conflitos de rótulo: 43 textos |
| dedup aproximado | 121.544 | 130.685 | 252.229 | 129 quase-duplicatas; 46 itens em conflito entre classes |
| exclusão do teste do corpus principal | 116.774 | 125.928 | 242.702 | 9527 textos (iguais ou quase iguais) |
| balanceamento (global) | 58.219 | 58.219 | 116.438 | descartados: 126264 |

## Atalhos: o quanto cada atributo prediz a classe

AUC com DD positivo; 0,5 = sem informação. Para o comprimento, o score é -palavras (AUC > 0,5 indica que frases curtas tendem a DD). Fonte e origem são medidas dentro de cada faixa de comprimento.

| conjunto | n | AUC comprimento | AUC fonte|faixa | AUC origem|faixa | mediana palavras DD/NDD |
|---|---|---|---|---|---|
| após filtros | 267.207 | 0.8622 | 0.7705 | 0.5000 | 7 / 11 |
| antes do balanceamento | 242.702 | 0.8601 | 0.7862 | 0.5000 | 7 / 11 |
| final | 116.438 | 0.8595 | 0.7858 | 0.5000 | 7 / 11 |
| final - train | 93.144 | 0.8595 | 0.7860 | 0.5000 | 7 / 11 |
| final - val | 11.664 | 0.8599 | 0.7839 | 0.5000 | 7 / 11 |
| final - test | 11.630 | 0.8592 | 0.7829 | 0.5000 | 7 / 11 |

## Histograma de comprimento (palavras)

| faixa | DD antes | NDD antes | % DD antes | por classe no final |
|---|---|---|---|---|
| 2 | 274 | 7 | 97.5% | 136 |
| 3 | 2.857 | 8 | 99.7% | 1.405 |
| 4 | 8.622 | 189 | 97.9% | 4.229 |
| 5 | 20.231 | 1.060 | 95.0% | 10.144 |
| 6 | 21.359 | 3.269 | 86.7% | 10.693 |
| 7-8 | 35.177 | 17.826 | 66.4% | 17.499 |
| 9-10 | 15.470 | 29.188 | 34.6% | 7.712 |
| 11-13 | 10.304 | 47.009 | 18.0% | 5.155 |
| 14-17 | 2.418 | 21.592 | 10.1% | 1.222 |
| 18-24 | 62 | 5.188 | 1.2% | 24 |
| 25-50 | 0 | 592 | 0.0% | 0 |

## Dedup aproximado

```
habilitado: True
pares_acima_do_limiar: 153
pares_entre_classes: 23
itens_em_conflito: 46
quase_duplicatas_removidas: 129
segundos: 3.9
```

## Composição final por origem

| origem|classe | n |
|---|---|
| legacy\|DD | 58.219 |
| legacy\|NDD | 58.219 |

## Composição final por fonte

| fonte|classe | n |
|---|---|
| glm\|DD | 6.850 |
| glm\|NDD | 6.760 |
| groq\|DD | 6.997 |
| groq\|NDD | 6.766 |
| lmstudio\|DD | 7.027 |
| lmstudio\|NDD | 6.846 |
| mistral\|DD | 2.224 |
| mistral\|NDD | 2.228 |
| nvidia\|DD | 4.557 |
| nvidia\|NDD | 4.110 |
| opencode\|DD | 2.836 |
| opencode\|NDD | 2.736 |
| openrouter\|DD | 6.035 |
| openrouter\|NDD | 4.825 |
| qwen\|DD | 14.781 |
| qwen\|NDD | 17.504 |
| sonnet\|DD | 6.912 |
| sonnet\|NDD | 6.444 |

## Partições

| partição | n |
|---|---|
| train | 93.144 |
| val | 11.664 |
| test | 11.630 |

Desvio máximo da participação de cada fonte entre partições: 0.035 p.p.
