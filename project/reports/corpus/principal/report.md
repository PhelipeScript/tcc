# Relatório de construção do corpus DDSD

Gerado em 2026-11-01T22:03:29-0300 em 8.4 s. Configuração: `config/corpus.toml`.

## Funil

| etapa | DD | NDD | total | nota |
|---|---|---|---|---|
| fontes | 147.421 | 142.499 | 289.920 | descartadas na leitura: {'rotulo_divergente': 0, 'texto_ausente': 0} |
| normalização e filtros | 147.344 | 142.389 | 289.733 | descartes: {'tamanho': 63, 'nao_ascii': 54, 'lista_serializada': 70} |
| dedup exato + conflitos | 132.824 | 141.814 | 274.638 | duplicatas: 15007; conflitos de rótulo: 44 textos |
| dedup aproximado | 132.707 | 141.746 | 274.453 | 139 quase-duplicatas; 46 itens em conflito entre classes |
| balanceamento (faixa_fonte_origem) | 58.219 | 58.219 | 116.438 | descartados: 158015 |

## Atalhos: o quanto cada atributo prediz a classe

AUC com DD positivo; 0,5 = sem informação. Para o comprimento, o score é -palavras (AUC > 0,5 indica que frases curtas tendem a DD). Fonte e origem são medidas dentro de cada faixa de comprimento.

| conjunto | n | AUC comprimento | AUC fonte|faixa | AUC origem|faixa | mediana palavras DD/NDD |
|---|---|---|---|---|---|
| após filtros | 289.733 | 0.8176 | 0.7755 | 0.5947 | 6 / 11 |
| antes do balanceamento | 274.453 | 0.8068 | 0.7743 | 0.5853 | 7 / 11 |
| final | 116.438 | 0.5150 | 0.5000 | 0.5000 | 9 / 9 |
| final - train | 93.150 | 0.5148 | 0.5001 | 0.5001 | 9 / 9 |
| final - val | 11.672 | 0.5164 | 0.5005 | 0.5008 | 9 / 9 |
| final - test | 11.616 | 0.5153 | 0.5005 | 0.5014 | 9 / 9 |

## Histograma de comprimento (palavras)

| faixa | DD antes | NDD antes | % DD antes | por classe no final |
|---|---|---|---|---|
| 2 | 2.031 | 1.511 | 57.3% | 1.502 |
| 3 | 4.649 | 1.816 | 71.9% | 1.751 |
| 4 | 10.475 | 2.062 | 83.6% | 1.964 |
| 5 | 22.294 | 3.137 | 87.7% | 2.961 |
| 6 | 23.397 | 5.309 | 81.5% | 5.287 |
| 7-8 | 36.717 | 19.368 | 65.5% | 15.445 |
| 9-10 | 16.765 | 30.487 | 35.5% | 13.007 |
| 11-13 | 12.164 | 48.885 | 19.9% | 12.164 |
| 14-17 | 3.406 | 22.691 | 13.1% | 3.378 |
| 18-24 | 809 | 5.888 | 12.1% | 760 |
| 25-50 | 0 | 592 | 0.0% | 0 |

## Dedup aproximado

```
habilitado: True
pares_acima_do_limiar: 163
pares_entre_classes: 23
itens_em_conflito: 46
quase_duplicatas_removidas: 139
segundos: 4.4
```

## Composição final por origem

| origem|classe | n |
|---|---|
| legacy\|DD | 47.695 |
| legacy\|NDD | 47.695 |
| mitigation\|DD | 10.524 |
| mitigation\|NDD | 10.524 |

## Composição final por fonte

| fonte|classe | n |
|---|---|
| claude_negdif\|DD | 1.211 |
| claude_negdif\|NDD | 1.211 |
| claude_subagentes\|DD | 361 |
| claude_subagentes\|NDD | 361 |
| gemini35flash\|DD | 27 |
| gemini35flash\|NDD | 27 |
| glm\|DD | 2.608 |
| glm\|NDD | 2.608 |
| gptoss120b\|DD | 703 |
| gptoss120b\|NDD | 703 |
| gptoss120b_negdif\|DD | 916 |
| gptoss120b_negdif\|NDD | 916 |
| groq\|DD | 3.088 |
| groq\|NDD | 3.088 |
| lmstudio\|DD | 5.955 |
| lmstudio\|NDD | 5.955 |
| lmstudio_negdif\|DD | 1.425 |
| lmstudio_negdif\|NDD | 1.425 |
| mistral\|DD | 1.829 |
| mistral\|NDD | 1.829 |
| nvidia\|DD | 1.681 |
| nvidia\|NDD | 1.681 |
| opencode\|DD | 2.342 |
| opencode\|NDD | 2.342 |
| openrouter\|DD | 3.369 |
| openrouter\|NDD | 3.369 |
| qwen\|DD | 20.415 |
| qwen\|NDD | 20.415 |
| qwen38\|DD | 914 |
| qwen38\|NDD | 914 |
| qwen38_negdif\|DD | 985 |
| qwen38_negdif\|NDD | 985 |
| sonnet\|DD | 10.390 |
| sonnet\|NDD | 10.390 |

## Partições

| partição | n |
|---|---|
| train | 93.150 |
| val | 11.672 |
| test | 11.616 |

Desvio máximo da participação de cada fonte entre partições: 0.068 p.p.
