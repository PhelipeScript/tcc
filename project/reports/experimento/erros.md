# Matriz de confusão e erros no teste interno

Linhas: rótulo do corpus; colunas: decisão do modelo (semente escolhida pela validação).

| modelo | n | DD → DD | DD → NDD (pedido perdido) | NDD → DD (falsa ativação) | NDD → NDD |
|---|---|---|---|---|---|
| debertinha-s44 | 11.616 | 5.705 | 103 | 164 | 5.644 |
| bertimbau-s44 | 11.616 | 5.719 | 89 | 106 | 5.702 |
| albertina-s42 | 11.616 | 5.706 | 102 | 130 | 5.678 |
| baseline_logreg | 11.616 | 5.586 | 222 | 232 | 5.576 |
| baseline_svm | 11.616 | 5.616 | 192 | 255 | 5.553 |

Erros mais confiantes do bertimbau-s44 (p(DD) do modelo). O teste mede a concordância com o rótulo do LLM gerador: um erro confiante pode ser um rótulo errado no corpus.

### Rotulados NDD, decididos DD

| frase | p(DD) | fonte | faixa |
|---|---|---|---|
| volta o video quero ver de novo aquela parte engracada | 0.9999 | groq | 9-10 |
| insere um sumario rapido antes da reuniao comecar | 0.9999 | qwen | 7-8 |
| compra um mouse com taxa de polling ajustavel por software | 0.9999 | qwen | 9-10 |
| cria esse atalho aqui na area de trabalho por favor | 0.9998 | qwen | 9-10 |
| tira print dessa tela antes que a promocao suma amor | 0.9997 | nvidia | 9-10 |
| meu pc ta estranho mostra os processos em execucao pra eu entender o que ta acontecendo | 0.9996 | qwen | 14-17 |
| manda essa noticia pro grupo agora e sobre a nova law de cinema | 0.9996 | glm | 11-13 |
| aplica o formato de porcentagem no relatorio | 0.9995 | qwen | 7-8 |
| muda a fonte do titulo | 0.9995 | qwen | 5 |
| cola isso no grupo do trabalho por favor | 0.9993 | openrouter | 7-8 |

### Rotulados DD, decididos NDD

| frase | p(DD) | fonte | faixa |
|---|---|---|---|
| pesquiso as notas de corte aqui | 0.0001 | qwen38 | 6 |
| ta tudo travado porque eu to tentando ditar e o sistema nao responde | 0.0001 | qwen | 11-13 |
| nao quero ver essa bagunca so o drive | 0.0008 | lmstudio | 7-8 |
| extrai os dados desse slide pra mim | 0.0010 | qwen | 7-8 |
| vamos mudar o alvo para aquela coordenada por favor | 0.0011 | lmstudio | 9-10 |
| baixa a mao na reuniao | 0.0011 | opencode | 5 |
| por que o som sumiu de repente no meio da chamada | 0.0011 | sonnet | 11-13 |
| olha eu to exausta mas preciso guardar a imagem dessa janela de estudo pra depois | 0.0012 | qwen38 | 14-17 |
| puxa o edge | 0.0013 | qwen38_negdif | 3 |
| rola pra baixo ai | 0.0014 | claude_negdif | 4 |
