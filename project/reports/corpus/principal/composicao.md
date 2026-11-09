# Composição do corpus final

Gerado por `scripts/relatorio_corpus.py` a partir das partições congeladas (SHA-256 conferido com `manifest.json`). Texto normalizado; tamanho em palavras.

## Por origem, modelo gerador e estratégia

| Origem | Modelo gerador | Estratégia | DD | NDD | Parte do corpus |
|---|---|---|---|---|---|
| Corpus inicial | qwen/qwen3.6-35b-a3b | - | 20.415 | 20.415 | 35,1% |
| Corpus inicial | claude-sonnet (CLI) | - | 10.390 | 10.390 | 17,8% |
| Corpus inicial | openrouter/free | - | 3.369 | 3.369 | 5,8% |
| Corpus inicial | qwen/qwen3.8-27b | - | 3.088 | 3.088 | 5,3% |
| Corpus inicial | zai-org/glm-4.6v-flash | - | 2.608 | 2.608 | 4,5% |
| Corpus inicial | opencode (CLI) | - | 2.342 | 2.342 | 4,0% |
| Corpus inicial | google/gemma-4-e4b | - | 1.973 | 1.973 | 3,4% |
| Corpus inicial | mistral-small-latest | - | 1.829 | 1.829 | 3,1% |
| Corpus inicial | moonshotai/kimi-k3 | - | 1.681 | 1.681 | 2,9% |
| Frases novas da correção | google/gemma-4-e4b | geração por tamanho | 3.982 | 3.982 | 6,8% |
| Frases novas da correção | google/gemma-4-e4b | negativos difíceis | 1.425 | 1.425 | 2,4% |
| Frases novas da correção | claude-sonnet-5-5 | negativos difíceis | 1.211 | 1.211 | 2,1% |
| Frases novas da correção | qwen/qwen3.8-27b | negativos difíceis | 985 | 985 | 1,7% |
| Frases novas da correção | openai/gpt-oss-120b | negativos difíceis | 916 | 916 | 1,6% |
| Frases novas da correção | qwen/qwen3.8-27b | geração por tamanho | 914 | 914 | 1,6% |
| Frases novas da correção | openai/gpt-oss-120b | geração por tamanho | 703 | 703 | 1,2% |
| Frases novas da correção | claude-sonnet-5-5 | geração por tamanho | 361 | 361 | 0,6% |
| Frases novas da correção | gemini-3.5-flash | geração por tamanho | 27 | 27 | < 0,1% |
| **Corpus inicial** | | | **47.695** | **47.695** | **81,9%** |
| **Frases novas da correção** | | | **10.524** | **10.524** | **18,1%** |

Nas frases novas: google/gemma-4-e4b, 51,4%; qwen/qwen3.8-27b, 18,0%; openai/gpt-oss-120b, 15,4%; claude-sonnet-5-5, 14,9%; gemini-3.5-flash, 0,3%.

## Partições

| Partição | DD | NDD | Total |
|---|---|---|---|
| Treino | 46.575 | 46.575 | 93.150 |
| Validação | 5.836 | 5.836 | 11.672 |
| Teste | 5.808 | 5.808 | 11.616 |

## Tamanho das frases (palavras)

| Classe | Mediana | 10% até | 90% até | Mínimo | Máximo |
|---|---|---|---|---|---|
| DD | 9 | 5 | 13 | 2 | 24 |
| NDD | 9 | 5 | 13 | 2 | 24 |

AUC com que só o tamanho prevê a classe: 0,818 depois dos filtros e 0,515 no corpus final.
