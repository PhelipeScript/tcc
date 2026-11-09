# Sessão ao vivo com roteiro - 20261106-101231

Sessão de 2026-11-06, das 10:12 às 10:22 (9,5 min), com o sistema final (`ddsd run --gravar`), o classificador `bertimbau-s44` (limiar 0,685) e o Whisper `mlx-community/whisper-large-v3-turbo` (mlx, semente 0). Roteiro: `docs/roteiro_sessao_ao_vivo.md`. Pontuação: `ddsd eval sessao`.

## Protocolo realizado

- Lidas na sessão: só as linhas de "você" no roteiro, 88 das 113 (41 DD e 47 NDD).
- As 25 falas das outras pessoas (Lucas, irmão, mãe) não foram lidas. Falas dirigidas ao computador entre elas, não testadas: 55.
- Caso difícil "pedidos a pessoas com vocabulário de computador": 5 de 11 linhas lidas (não lidas: 23, 28, 32, 36, 49, 85).
- Caso difícil "o mesmo pedido feito primeiro a uma pessoa e depois ao computador": 3 de 6 linhas lidas (não lidas: 28, 36, 53).
- Caso difícil "falas de até 4 palavras nas duas classes": 13 de 14 linhas lidas (não lidas: 38).

## Casamento com o roteiro

- 92 falas detectadas para 88 linhas lidas: 84 casadas uma a uma, 2 junções (08 + 09; 108 + 109), nenhuma divisão, nenhuma linha sem fala e 6 falas fora do roteiro.
- Menor similaridade aceita num casamento: 0,64 (linha 50); o limiar é 0,50.

## Resultado

| medida | no texto do roteiro | na sessão (transcrição do Whisper) |
|---|---|---|
| Pedidos atendidos (DD) | 40 de 41 (97,6%) | 36 de 41 (87,8%) |
| Falsas ativações (NDD) | 6 de 45 (13,3%) | 4 de 45 (8,9%) |

Unidade: o grupo casado (as falas que o sistema decidiu, com as linhas do roteiro dentro delas). O grupo é DD se alguma linha dele for DD, e é ativado se alguma fala dele foi decidida DD. "No texto do roteiro" é o mesmo classificador julgando o texto que deveria ter sido lido. As falas fora do roteiro não entram nas taxas.

No texto do roteiro, o classificador perderia o pedido 92 e ativaria nas linhas NDD 24, 50, 84, 86, 97, 104.

Contando por linha, as linhas NDD dentro de falas ativadas são 5 de 47 (08, 50, 86, 97, 104), porque a linha 08 foi dita na mesma fala que um pedido ao computador.

## Erros e origem

| linha | quem → para quem | roteiro | transcrição | decisão (p DD) | no roteiro (p DD) | origem | áudio |
|---|---|---|---|---|---|---|---|
| 41 | você → computador | Manda uma mensagem pro professor perguntando se a reunião de sexta continua de pé. | Hum, mandou uma mensagem pro professor perguntando se a reunião de sexta continua de pé. | NDD (0,00) | DD (1,00) | transcricao | U0033.wav |
| 50 | você → mãe | Abro sim. | Abro sim, calma aí. | DD (0,99) | DD (0,99) | classificador | U0037.wav |
| 86 | você → irmão | Posso, me manda o arquivo por e-mail. | Posso. Me manda um arquivo por e-mail. | DD (1,00) | DD (1,00) | classificador | U0067.wav |
| 92 | você → computador | Imprime de novo, só que frente e verso. | imprime de novo, só que frente inverso | NDD (0,02) | NDD (0,03) | classificador | U0071.wav |
| 97 | você → Fernanda | Manda de novo pra mim, por favor, no endereço da faculdade. | Manda de novo pra mim, por favor, no endereço da faculdade. | DD (0,97) | DD (0,97) | classificador | U0076.wav |
| 103 | você → computador | Abre a apresentação que a Fernanda mandou. | Abri a apresentação que a Fernanda mandou. | NDD (0,00) | DD (1,00) | transcricao | U0084.wav |
| 104 | você → Fernanda | Abre aí o slide sete e aumenta a fonte do gráfico, que tá muito pequena. | Abra aí o slide set e aumenta a fonte do gráfico, que está muito pequena. | DD (1,00) | DD (0,99) | classificador | U0085.wav |
| 106 | você → computador | Cria um lembrete pra amanhã às nove da manhã pra eu mandar o capítulo pro meu orientador. | Eu não lembrei de ir para amanhã às 9 da manhã para eu mandar o capítulo para o meu orientador. | NDD (0,00) | DD (1,00) | transcricao | U0087.wav |
| 113 | você → computador | Bloqueia a tela que eu vou almoçar. | Bloqueei a tela que eu vou almoçar. | NDD (0,00) | DD (1,00) | transcricao | U0092.wav |

- Pedidos perdidos (DD): 5; segmentacao 0, classificador 1, transcricao 4.
- Falsas ativações (NDD): 4; segmentacao 0, classificador 4, transcricao 0.
- `classificador`: o modelo erra também no texto do roteiro. `transcricao`: acerta no texto do roteiro e erra na transcrição. Sem uma transcrição manual, essa origem junta o erro do ASR e o desvio da leitura (improviso ou palavra trocada); para separar, é preciso ouvir os áudios listados, em `data/sessoes/20261106-101231/`. `segmentacao`: junção, divisão ou linha sem fala.

## Recortes

Por destinatário da fala (grupos casados):

| destinatário | grupos | ativados | taxa |
|---|---|---|---|
| computador | 41 | 36 | 87,8% |
| pensando alto | 13 | 0 | 0,0% |
| pessoa no quarto | 14 | 2 | 14,3% |
| pessoa na chamada | 18 | 2 | 11,1% |

Casos difíceis do roteiro, só nas linhas lidas:

| caso | linhas lidas | DD: pedidos atendidos | NDD: falsas ativações |
|---|---|---|---|
| pedidos a pessoas com vocabulário de computador | 5 de 11 | - | 3 de 5 (60,0%) |
| o mesmo pedido feito primeiro a uma pessoa e depois ao computador | 3 de 6 | 2 de 3 (66,7%) | - |
| comentários sobre o computador ou a impressora | 4 de 4 | - | 0 de 4 (0,0%) |
| falas de até 4 palavras nas duas classes | 13 de 14 | 5 de 5 (100,0%) | 1 de 8 (12,5%) |

Por cena:

| cena | pedidos atendidos | falsas ativações |
|---|---|---|
| Cena 1 - Começando o dia | 11 de 11 (100,0%) | 0 de 6 (0,0%) |
| Cena 2 - Chamada de vídeo com o Lucas, colega de TCC | 5 de 6 (83,3%) | 0 de 9 (0,0%) |
| Cena 3 - Sua mãe entra no quarto | 2 de 2 (100,0%) | 1 de 7 (14,3%) |
| Cena 4 - De volta ao trabalho | 13 de 13 (100,0%) | 0 de 5 (0,0%) |
| Cena 5 - Seu irmão entra no quarto | 2 de 3 (66,7%) | 1 de 7 (14,3%) |
| Cena 6 - Ligação no celular | 1 de 2 (50,0%) | 2 de 9 (22,2%) |
| Cena 7 - Pausa para o almoço | 2 de 4 (50,0%) | 0 de 2 (0,0%) |

## Falas fora do roteiro

6 falas sem linha no roteiro (improvisos de quem leu); 0 delas ativaram o sistema.

| fala | cena | transcrição | decisão (p DD) | confiabilidade |
|---|---|---|---|---|
| U0018 | Cena 2 - Chamada de vídeo com o Lucas, colega de TCC | Aí eu preciso ligar pro Lucas, calma aí. | NDD (0,01) | alta |
| U0043 | Cena 4 - De volta ao trabalho | Vamos voltar ao trabalho. | NDD (0,00) | média |
| U0062 | Cena 5 - Seu irmão entra no quarto | Hum? | NDD (0,00) | média |
| U0073 | Cena 6 - Ligação no celular | Oi gente Fernanda tá me ligando | NDD (0,00) | baixa |
| U0078 | Cena 6 - Ligação no celular | Uhum. | NDD (0,00) | média |
| U0079 | Cena 6 - Ligação no celular | Uhum. | NDD (0,00) | média |

## Transcrição

- WER contra o roteiro: 11,3% (10,5% sem as pausas preenchidas: ah, ahm, ahn, eh, ehm, hm, hum, uh, uhm), em 86 grupos casados.
- O WER mede a transcrição contra o texto do roteiro. Por isso inclui os desvios da própria leitura (improvisos e palavras trocadas), além dos erros do ASR.

## Tempo de resposta

| etapa | falas | p50 | p95 | máximo |
|---|---|---|---|---|
| Fim da fala → decisão (inclui a espera de 0,7 s do detector de voz) | 92 | 1,161 s | 1,227 s | 1,241 s |
| Whisper | 92 | 0,410 s | 0,468 s | 0,501 s |
| Classificador | 92 | 0,046 s | 0,053 s | 0,057 s |

## Verificações

- Paridade: julgando de novo as 92 transcrições, o classificador repetiu 92 decisões (max |Δp| 0).
- Classificador ativo em todas as falas: sim (`bertimbau`). Motivos das decisões: classificador 92. Fechamento das falas: fim_da_fala 92. Prazo de segurança estourado: 0.
- Áudio: 92 arquivos WAV, 92 com a duração do trecho da decisão (±0,05 s).
- Gravador: 0 erros de gravação, 0 pendentes no fim, thread encerrada.
- Transbordamento da captura: não registrado (a gravação da sessão não guarda o contador do painel).
- Confiabilidade das transcrições: alta 59, média 32, baixa 1.

## Todas as falas

| linha | fala | quem → para quem | roteiro | transcrição | decisão (p DD) | certo |
|---|---|---|---|---|---|---|
| 01 | U0001 | você → computador | Abre o navegador e entra no meu e-mail da faculdade. | Abra o navegador e entra no meu e-mail da faculdade. | DD (1,00) | sim |
| 02 | U0002 | você → (pensando alto) | Nossa, quarenta e-mails sem ler, não acredito. | Nossa, 40,5 sem ler. Não acredito. | NDD (0,00) | sim |
| 03 | U0003 | você → computador | Procura o e-mail do meu orientador com a correção do capítulo dois. | Procura o e-mail do meu orientador com a correção do capítulo 2. | DD (1,00) | sim |
| 04 | U0004 | você → (pensando alto) | Ele marcou um monte de coisa em vermelho de novo. | Ele marcou um monte de coisa em vermelho de novo | NDD (0,00) | sim |
| 05 | U0005 | você → computador | Baixa esse anexo e abre no editor de texto. | Baixe esse anexo e abra o editor de texto | DD (1,00) | sim |
| 06 | U0006 | você → computador | Coloca a minha playlist de estudo pra tocar bem baixinho. | Coloca a minha playlist de estudo pra tocar bem baixinho. | DD (1,00) | sim |
| 07 | U0007 | você → computador | Aumenta só um pouquinho o volume. | Aumenta só um pouquinho o volume. | DD (1,00) | sim |
| 08 + 09 | U0008 | você → (pensando alto) + você → computador | Agora sim, dá pra concentrar. Abre a agenda e mostra o que eu tenho pra hoje. | Agora sim, dá pra concentrar. Abre a agenda e mostra o que eu tenho pra hoje. | DD (1,00) | sim |
| 10 | U0009 | você → (pensando alto) | Reunião só às quatro, então dá tempo de revisar tudo. | Hum, reunião só às quatro? Então dá tempo de revisar tudo. | NDD (0,00) | sim |
| 11 | U0010 | você → computador | Cria uma nota nova chamada pendências do TCC. | Crie uma nota nova chamada pendências do TCC. | DD (1,00) | sim |
| 12 | U0011 | você → computador | Escreve revisar a figura cinco e conferir as referências. | Escreve revisar a figura 5 e conferir as referências. | DD (1,00) | sim |
| 13 | U0012 | você → (pensando alto) | Onde foi que eu salvei aquele artigo do Whisper, meu Deus? | Onde foi que eu salvei aquele artigo do Whisper? Meu Deus. | NDD (0,40) | sim |
| 14 | U0013 | você → computador | Procura nos downloads um arquivo com o nome Whisper. | Procuremos downloads no arquivo com o nome Whisper. | DD (1,00) | sim |
| 15 | U0014 | você → computador | Abre esse PDF em tela cheia. | Abre esse PDF que ela cheia. | DD (1,00) | sim |
| 16 | U0015 | você → (pensando alto) | Achei, era esse mesmo. | É, eu achei que era esse mesmo. | NDD (0,00) | sim |
| 17 | U0016 | você → computador | Grifa esse parágrafo de amarelo. | Grifa esse parágrafo de amarelo. | DD (1,00) | sim |
| 18 | U0017 | você → (pensando alto) | Isso aqui vai direto pra introdução. | Isso aqui vai direto pra introdução. | NDD (0,01) | sim |
| - | U0018 | fora do roteiro | - | Aí eu preciso ligar pro Lucas, calma aí. | NDD (0,01) | - |
| 19 | U0019 | você → Lucas | E aí, Lucas, tudo certo? | E aí, Lucas, tudo certo? | NDD (0,00) | sim |
| 21 | U0020 | você → Lucas | Consegui, mas demorou umas duas horas pra terminar tudo. | eu consegui mas demorou umas duas horas para terminar tudo | NDD (0,00) | sim |
| 22 | U0021 | você → computador | Pausa essa música rapidinho. | pausa essa música rapidinho | DD (1,00) | sim |
| 24 | U0022 | você → Lucas | Consigo, só um minutinho que eu vou abrir aqui. | Eu consigo, só um minutinho que eu vou abrir aqui. | NDD (0,00) | sim |
| 25 | U0023 | você → computador | Abre a pasta de figuras do projeto. | Abre a pasta de figuras do projeto. | DD (1,00) | sim |
| 26 | U0024 | você → computador | Muda para a janela da chamada. | Muda para a janela da chamada. | DD (1,00) | sim |
| 27 | U0025 | você → Lucas | Tá vendo essa curva aqui embaixo? | Tá vendo essa curva aqui embaixo? | NDD (0,01) | sim |
| 29 | U0026 | você → computador | Amplia a parte de baixo desse gráfico. | Amplie a parte de baixo desse gráfico. | DD (0,98) | sim |
| 31 | U0027 | você → Lucas | O modelo fica confiante demais nos erros, mas a F1 continua melhorando. | O modelo fica confiante demais nos rios, mas a F1 continua melhorando. | NDD (0,00) | sim |
| 33 | U0028 | você → Lucas | Mando sim, assim que eu terminar de revisar. | Mando sim, assim que eu terminar de revisar. | NDD (0,01) | sim |
| 35 | U0029 | você → Lucas | Já anotei aqui. | Já notei aqui. | NDD (0,00) | sim |
| 37 | U0030 | você → Lucas | Deixa comigo, eu mando ainda hoje. | Deixa comigo, eu mando ainda hoje. | NDD (0,00) | sim |
| 39 | U0031 | você → Lucas | Falou, até amanhã! | Falou, até amanhã. | NDD (0,00) | sim |
| 40 | U0032 | você → computador | Fecha a chamada e volta a tocar a música. | Fecha a chamada e volta a tocar a música. | DD (1,00) | sim |
| 41 | U0033 | você → computador | Manda uma mensagem pro professor perguntando se a reunião de sexta continua de pé. | Hum, mandou uma mensagem pro professor perguntando se a reunião de sexta continua de pé. | NDD (0,00) | **não** |
| 43 | U0034 | você → mãe | Tá bom, mãe, já vou. | Tá bom, mãe? Já vou. | NDD (0,00) | sim |
| 45 | U0035 | você → mãe | Suco, por favor. | Suco, por favor. | NDD (0,00) | sim |
| 47 | U0036 | você → mãe | Acho que vem, mas só depois do almoço. | Acho que vem, mas só depois do almoço. | NDD (0,00) | sim |
| 50 | U0037 | você → mãe | Abro sim. | Abro sim, calma aí. | DD (0,99) | **não** |
| 52 | U0038 | você → mãe | É o meu TCC, o computador entende quando eu falo com ele. | É o meu TCC. O computador entende quando eu falo com ele. | NDD (0,00) | sim |
| 54 | U0039 | você → computador | Toca uma música do Roberto Carlos. | Toca uma música do Roberto Carlos. | DD (1,00) | sim |
| 57 | U0040 | você → mãe | Agora deixa eu terminar aqui, senão eu não acabo nunca. | Agora deixa eu terminar aqui, senão eu não acabo nunca. | NDD (0,00) | sim |
| 58 | U0041 | você → computador | Para a música. | Para música. | DD (1,00) | sim |
| 60 | U0042 | você → mãe | Pode deixar, encosta a porta quando sair, por favor. | Pode deixar. Encosta a porta quando sair, por favor. | NDD (0,00) | sim |
| - | U0043 | fora do roteiro | - | Vamos voltar ao trabalho. | NDD (0,00) | - |
| 61 | U0044 | você → computador | Volta para o documento do capítulo três. | Volta para o documento do capítulo 3. | DD (1,00) | sim |
| 62 | U0045 | você → computador | Procura no texto onde aparece a palavra limiar. | Procura no texto onde aparece a palavra limiar. | DD (1,00) | sim |
| 63 | U0046 | você → (pensando alto) | Opa, aqui tá errado, não foi esse o limiar que a gente usou. | Opa, aqui tá errado. Não foi esse limiar que a gente usou. | NDD (0,00) | sim |
| 64 | U0047 | você → computador | Apaga esse parágrafo inteiro. | Apague esse parágrafo inteiro. | DD (1,00) | sim |
| 65 | U0048 | você → computador | Escreve o limiar foi escolhido na validação e o teste foi avaliado uma única vez. | Escreve o limiar, foi escolhido na validação e o teste foi avaliado uma única vez. | DD (1,00) | sim |
| 66 | U0049 | você → computador | Lê em voz alta o último parágrafo. | Língua exalta o último parágrafo. | DD (0,93) | sim |
| 67 | U0050 | você → (pensando alto) | Ficou bom, mas dá pra melhorar o final. | Uhum, ficou bom. Mas dá pra melhorar até o final. | NDD (0,00) | sim |
| 68 | U0051 | você → computador | Salva o documento. | Salve o documento. | DD (1,00) | sim |
| 69 | U0052 | você → computador | Converte o capítulo para PDF e salva na pasta do TCC. | Converte o capítulo para PDF e salva na pasta do TCC. | DD (1,00) | sim |
| 70 | U0053 | você → (pensando alto) | Ufa, agora só falta a parte do sistema ao vivo. | Ufa! Agora só falta a parte do sistema ao vivo. | NDD (0,00) | sim |
| 71 | U0054 | você → computador | Tira um print dessa tela e salva na área de trabalho. | Tira um print dessa tela e salva na área de trabalho. | DD (1,00) | sim |
| 72 | U0055 | você → computador | Abre uma nova aba e pesquisa como citar um site nas normas da ABNT. | Abre uma nova aba e pesquisa como citar um site nas normas da ABNT. | DD (1,00) | sim |
| 73 | U0056 | você → computador | Entra no primeiro resultado. | Entra no primeiro resultado. | DD (1,00) | sim |
| 74 | U0057 | você → (pensando alto) | Esse site tá cheio de propaganda, que coisa chata. | Esse site tá cheio de propaganda, coisa chata. | NDD (0,00) | sim |
| 75 | U0058 | você → computador | Volta para a página anterior. | Bom, volta para a página anterior. | DD (1,00) | sim |
| 76 | U0059 | você → computador | Diminui o brilho da tela, que tá machucando o olho. | Diminui o brilho da tela, que tá machucando meu olho. | DD (1,00) | sim |
| 77 | U0060 | você → (pensando alto) | Esse computador tá esquentando demais hoje. | Hum, esse computador tá esquentando demais hoje. | NDD (0,00) | sim |
| 78 | U0061 | você → computador | Fecha os programas que eu não tô usando. | Fecha os programas que eu não estou usando. | DD (1,00) | sim |
| - | U0062 | fora do roteiro | - | Hum? | NDD (0,00) | - |
| 80 | U0063 | você → irmão | Tá ali em cima da escrivaninha, do lado da impressora. | Ah, tá ali em cima da escrivaninha, do lado da impressora. | NDD (0,00) | sim |
| 82 | U0064 | você → irmão | Pera, que eu vou ver aqui pra você. | Hum, peraí que eu vou ver aqui pra você. | NDD (0,00) | sim |
| 83 | U0065 | você → computador | Mostra a senha da rede Wi-Fi. | Mostra assim a da rede Wi-Fi. | DD (1,00) | sim |
| 84 | U0066 | você → irmão | Anota aí, tá aparecendo na tela. | A nota aí, está aparecendo na tela. | NDD (0,00) | sim |
| 86 | U0067 | você → irmão | Posso, me manda o arquivo por e-mail. | Posso. Me manda um arquivo por e-mail. | DD (1,00) | **não** |
| 88 | U0068 | você → computador | Imprime o anexo do último e-mail. | imprime o anexo do último e-mail. | DD (1,00) | sim |
| 89 | U0069 | você → (pensando alto) | Essa impressora sempre demora pra começar. | Essa impressora sempre demora pra começar. | NDD (0,00) | sim |
| 90 | U0070 | você → irmão | Ih, saiu só de um lado, a escola pediu frente e verso? | Isso aí só de um lado? A escola pediu frente e verso? | NDD (0,00) | sim |
| 92 | U0071 | você → computador | Imprime de novo, só que frente e verso. | imprime de novo, só que frente inverso | NDD (0,02) | **não** |
| 94 | U0072 | você → irmão | De nada, depois você me paga um sorvete. | De nada. Depois você me paga um sorvete. | NDD (0,00) | sim |
| - | U0073 | fora do roteiro | - | Oi gente Fernanda tá me ligando | NDD (0,00) | - |
| 95 | U0074 | você → Fernanda | Oi, Fernanda, pode falar. | Oi, Fernando. Pode falar. | NDD (0,01) | sim |
| 96 | U0075 | você → Fernanda | Não, eu ainda não recebi o seu e-mail. | Não, ainda não recebi seu e-mail. | NDD (0,00) | sim |
| 97 | U0076 | você → Fernanda | Manda de novo pra mim, por favor, no endereço da faculdade. | Manda de novo pra mim, por favor, no endereço da faculdade. | DD (0,97) | **não** |
| 98 | U0077 | você → Fernanda | Isso, e coloca o arquivo em PDF, que fica mais fácil de abrir. | Isso. E coloca o arquivo em PDF que fica mais fácil de abrir. | NDD (0,54) | sim |
| - | U0078 | fora do roteiro | - | Uhum. | NDD (0,00) | - |
| - | U0079 | fora do roteiro | - | Uhum. | NDD (0,00) | - |
| 99 | U0080 | você → Fernanda | Pera aí, deixa eu ver se chegou. | Pera aí, deixa eu ver se chegou. | NDD (0,00) | sim |
| 100 | U0081 | você → computador | Atualiza a caixa de entrada. | Atualiza a caixa de entrada. | DD (0,98) | sim |
| 101 | U0082 | você → Fernanda | Chegou, tá aqui, valeu! | Chegou, tá aqui, valeu. | NDD (0,00) | sim |
| 102 | U0083 | você → Fernanda | Tô abrindo aqui, só um segundo. | Tô abrindo aqui, só um segundo. | NDD (0,00) | sim |
| 103 | U0084 | você → computador | Abre a apresentação que a Fernanda mandou. | Abri a apresentação que a Fernanda mandou. | NDD (0,00) | **não** |
| 104 | U0085 | você → Fernanda | Abre aí o slide sete e aumenta a fonte do gráfico, que tá muito pequena. | Abra aí o slide set e aumenta a fonte do gráfico, que está muito pequena. | DD (1,00) | **não** |
| 105 | U0086 | você → Fernanda | Beleza, depois a gente conversa, tchau! | beleza, depois a gente conversa tchau, tchau | NDD (0,00) | sim |
| 106 | U0087 | você → computador | Cria um lembrete pra amanhã às nove da manhã pra eu mandar o capítulo pro meu orientador. | Eu não lembrei de ir para amanhã às 9 da manhã para eu mandar o capítulo para o meu orientador. | NDD (0,00) | **não** |
| 107 | U0088 | você → computador | Fecha todas as abas do navegador. | Fecha todas as abas do navegador. | DD (1,00) | sim |
| 108 + 109 | U0089 | você → (pensando alto) + você → (pensando alto) | Chega por agora, minha cabeça já não funciona mais. Depois do almoço eu volto pra terminar a conclusão. | Ai, chega por agora, minha cabeça já não funciona mais. Depois do almoço eu volto pra terminar a conclusão. | NDD (0,00) | sim |
| 110 | U0090 | você → computador | Coloca no modo não perturbe. | Coloca no modo não perturbe. | DD (1,00) | sim |
| 112 | U0091 | você → mãe | Tô indo! | Tô indo. | NDD (0,00) | sim |
| 113 | U0092 | você → computador | Bloqueia a tela que eu vou almoçar. | Bloqueei a tela que eu vou almoçar. | NDD (0,00) | **não** |
