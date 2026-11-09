# Revisão humana cega - phelipe

300 de 300 frases anotadas. Concordância 0.937; kappa de Cohen 0.873.

| LLM \ humano | DD | NDD |
|---|---|---|
| DD | 137 | 13 |
| NDD | 6 | 144 |

### Por origem

| grupo | n | concordância | kappa |
|---|---|---|---|
| legacy | 182 | 0.9451 | 0.8901 |
| mitigation | 118 | 0.9237 | 0.8477 |

### Por fonte

| grupo | n | concordância | kappa |
|---|---|---|---|
| claude_negdif | 10 | 1.0000 | 1.0000 |
| claude_subagentes | 8 | 1.0000 | 1.0000 |
| gemini35flash | 6 | 1.0000 | 1.0000 |
| glm | 17 | 1.0000 | 1.0000 |
| gptoss120b | 16 | 0.9375 | 0.8750 |
| gptoss120b_negdif | 10 | 0.8000 | 0.6000 |
| groq | 23 | 0.9130 | 0.8271 |
| lmstudio | 32 | 0.9062 | 0.8125 |
| lmstudio_negdif | 12 | 0.9167 | 0.8333 |
| mistral | 15 | 0.9333 | 0.8673 |
| nvidia | 9 | 1.0000 | 1.0000 |
| opencode | 17 | 0.9412 | 0.8828 |
| openrouter | 20 | 1.0000 | 1.0000 |
| qwen | 42 | 0.9286 | 0.8571 |
| qwen38 | 16 | 1.0000 | 1.0000 |
| qwen38_negdif | 13 | 0.8462 | 0.6829 |
| sonnet | 34 | 0.9118 | 0.8172 |

### Por estratégia

| grupo | n | concordância | kappa |
|---|---|---|---|
| comprimento | 73 | 0.9452 | 0.8905 |
| legado | 182 | 0.9451 | 0.8901 |
| negativos_dificeis | 45 | 0.8889 | 0.7788 |

### Por faixa

| grupo | n | concordância | kappa |
|---|---|---|---|
| 11-13 | 44 | 0.9545 | 0.9076 |
| 14-17 | 26 | 0.9615 | 0.9231 |
| 18-24 | 12 | 1.0000 | 1.0000 |
| 2 | 25 | 0.8800 | 0.7492 |
| 3 | 25 | 0.8400 | 0.6774 |
| 4 | 29 | 0.8966 | 0.7924 |
| 5 | 30 | 0.9333 | 0.8667 |
| 6 | 31 | 0.9032 | 0.8058 |
| 7-8 | 39 | 0.9744 | 0.9475 |
| 9-10 | 39 | 1.0000 | 1.0000 |

### Divergências

| id | texto | LLM | humano | fonte | faixa |
|---|---|---|---|---|---|
| A0021 | volta rápido | DD | NDD | qwen38_negdif | 2 |
| A0030 | manda endereço do blog | DD | NDD | gptoss120b_negdif | 4 |
| A0041 | Devagar, deixa eu. | NDD | DD | lmstudio | 3 |
| A0057 | verificar a assinatura dele deu erro | NDD | DD | opencode | 6 |
| A0101 | Queria pular um trecho. | DD | NDD | lmstudio | 4 |
| A0108 | fecha essa janela da reunião que tá atrapalhando | NDD | DD | qwen | 7-8 |
| A0121 | imprime apenas o que eu selecionei | NDD | DD | qwen | 6 |
| A0128 | só mais uma música, ok | DD | NDD | gptoss120b | 5 |
| A0176 | Mostra ontem | DD | NDD | lmstudio | 2 |
| A0180 | Manda fechar essa porcaria de janela | DD | NDD | mistral | 6 |
| A0185 | verifica quantas flechas de treino já usei esse mês segundo o registro do aplicativo | DD | NDD | sonnet | 14-17 |
| A0199 | Minha planilha trava | DD | NDD | qwen | 3 |
| A0201 | compartilha o parecer da banca com o aluno depois da defesa | DD | NDD | sonnet | 11-13 |
| A0213 | que horas são já | NDD | DD | groq | 4 |
| A0261 | envia o link | DD | NDD | qwen38_negdif | 3 |
| A0264 | Dá toca mais | DD | NDD | lmstudio_negdif | 3 |
| A0270 | Tente desmutar o microfone novamente. | NDD | DD | groq | 5 |
| A0292 | dá página | DD | NDD | gptoss120b_negdif | 2 |
| A0298 | verifica o saldo devedor atualizado antes de decidir quitar o financiamento antecipado | DD | NDD | sonnet | 11-13 |
