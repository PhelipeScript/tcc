# Roteiro da sessão ao vivo

Sessão de cerca de 12 minutos com o sistema final (`ddsd run`), no microfone, simulando um dia de
trabalho no quarto: pedidos ao computador, uma chamada de vídeo com um colega, a mãe e o irmão que entram no
quarto e uma ligação no celular. Cada fala tem um número e um destinatário. Quando o destinatário é o
**computador**, a fala é dirigida a ele (DD); as demais são não dirigidas (NDD). Assim dá para conferir,
depois, cada decisão do sistema com o roteiro.

## Como começar

- Sala quieta, sem música nem vídeo tocando. O sistema só classifica, não executa os pedidos: nenhuma música vai
  tocar de verdade, e nenhum som deve ser tocado para simular.
- Microfone interno do MacBook.
- No seu terminal:

  ```
  uv run ddsd run --gravar
  ```

  O `--gravar` salva o áudio de cada fala e as evidências da decisão em `data/sessoes/<data-hora>/`. Espere o
  painel mostrar o estado ocioso antes da primeira fala. Para encerrar, tecle Q.

## Como ler

- Uma linha por vez, de uma vez só, sem pausa no meio: o detector de voz fecha a fala depois de 0,7 s de silêncio.
- Cerca de 2 s de silêncio entre uma linha e a próxima.
- A coluna "Quem → para quem" é só indicação de cena, não se lê em voz alta. Nas falas para o computador, fale olhando para a tela, como num pedido normal, sem chamar o computador pelo nome.
- As falas do Lucas, da sua mãe e do seu irmão também são lidas por você, com outra entonação, ou apenas ignoradas. Se alguém da casa puder ler as falas da mãe ou do irmão, melhor ainda.
- Se o sistema errar, não repita nem corrija a fala: siga o roteiro. Se você tropeçar numa palavra, siga em frente.
- Entre uma cena e outra, fique uns 15 s em silêncio, como se estivesse lendo a tela.

## Cena 1 - Começando o dia

*Você senta na frente do computador para trabalhar no TCC.*

| # | Quem → para quem | Fala |
|---|---|---|
| 01 | você → **computador** | Abre o navegador e entra no meu e-mail da faculdade. |
| 02 | você → (pensando alto) | Nossa, quarenta e-mails sem ler, não acredito. |
| 03 | você → **computador** | Procura o e-mail do meu orientador com a correção do capítulo dois. |
| 04 | você → (pensando alto) | Ele marcou um monte de coisa em vermelho de novo. |
| 05 | você → **computador** | Baixa esse anexo e abre no editor de texto. |
| 06 | você → **computador** | Coloca a minha playlist de estudo pra tocar bem baixinho. |
| 07 | você → **computador** | Aumenta só um pouquinho o volume. |
| 08 | você → (pensando alto) | Agora sim, dá pra concentrar. |
| 09 | você → **computador** | Abre a agenda e mostra o que eu tenho pra hoje. |
| 10 | você → (pensando alto) | Reunião só às quatro, então dá tempo de revisar tudo. |
| 11 | você → **computador** | Cria uma nota nova chamada pendências do TCC. |
| 12 | você → **computador** | Escreve revisar a figura cinco e conferir as referências. |
| 13 | você → (pensando alto) | Onde foi que eu salvei aquele artigo do Whisper, meu Deus? |
| 14 | você → **computador** | Procura nos downloads um arquivo com o nome Whisper. |
| 15 | você → **computador** | Abre esse PDF em tela cheia. |
| 16 | você → (pensando alto) | Achei, era esse mesmo. |
| 17 | você → **computador** | Grifa esse parágrafo de amarelo. |
| 18 | você → (pensando alto) | Isso aqui vai direto pra introdução. |

## Cena 2 - Chamada de vídeo com o Lucas, colega de TCC

*O Lucas liga pelo computador. A música do roteiro continua "tocando" até você pedir para pausar.*

| # | Quem → para quem | Fala |
|---|---|---|
| 19 | você → Lucas | E aí, Lucas, tudo certo? |
| 20 | Lucas → você | Tudo certo, conseguiu rodar aqueles testes ontem? |
| 21 | você → Lucas | Consegui, mas demorou umas duas horas pra terminar tudo. |
| 22 | você → **computador** | Pausa essa música rapidinho. |
| 23 | Lucas → você | Você consegue compartilhar a tela pra eu ver o gráfico? |
| 24 | você → Lucas | Consigo, só um minutinho que eu vou abrir aqui. |
| 25 | você → **computador** | Abre a pasta de figuras do projeto. |
| 26 | você → **computador** | Muda para a janela da chamada. |
| 27 | você → Lucas | Tá vendo essa curva aqui embaixo? |
| 28 | Lucas → você | Dá um zoom nessa parte, que daqui eu não tô enxergando nada. |
| 29 | você → **computador** | Amplia a parte de baixo desse gráfico. |
| 30 | Lucas → você | Agora deu pra ver, mas por que a perda sobe no final? |
| 31 | você → Lucas | O modelo fica confiante demais nos erros, mas a F1 continua melhorando. |
| 32 | Lucas → você | Faz sentido, me manda esse arquivo depois, por favor. |
| 33 | você → Lucas | Mando sim, assim que eu terminar de revisar. |
| 34 | Lucas → você | E não esquece de atualizar a tabela de resultados. |
| 35 | você → Lucas | Já anotei aqui. |
| 36 | Lucas → você | Ah, e manda uma mensagem pro professor confirmando a reunião de sexta. |
| 37 | você → Lucas | Deixa comigo, eu mando ainda hoje. |
| 38 | Lucas → você | Valeu, até mais! |
| 39 | você → Lucas | Falou, até amanhã! |
| 40 | você → **computador** | Fecha a chamada e volta a tocar a música. |
| 41 | você → **computador** | Manda uma mensagem pro professor perguntando se a reunião de sexta continua de pé. |

## Cena 3 - Sua mãe entra no quarto

*Ela abre a porta e fica um pouco conversando com você.*

| # | Quem → para quem | Fala |
|---|---|---|
| 42 | mãe → você | Ei, o almoço vai sair daqui a uns vinte minutos. |
| 43 | você → mãe | Tá bom, mãe, já vou. |
| 44 | mãe → você | Você vai querer suco ou refrigerante? |
| 45 | você → mãe | Suco, por favor. |
| 46 | mãe → você | E aquele seu amigo, o Lucas, vem estudar aqui amanhã? |
| 47 | você → mãe | Acho que vem, mas só depois do almoço. |
| 48 | mãe → você | Nossa, como esse quarto tá abafado. |
| 49 | mãe → você | Abre essa janela um pouquinho. |
| 50 | você → mãe | Abro sim. |
| 51 | mãe → você | O que você tanto faz nesse computador o dia inteiro? |
| 52 | você → mãe | É o meu TCC, o computador entende quando eu falo com ele. |
| 53 | mãe → você | Sério, então pede pra ele tocar uma música do Roberto Carlos. |
| 54 | você → **computador** | Toca uma música do Roberto Carlos. |
| 55 | mãe → **computador** | Aumenta esse volume, que eu quero ouvir. |
| 56 | mãe → você | Que coisa, no meu tempo não tinha nada disso. |
| 57 | você → mãe | Agora deixa eu terminar aqui, senão eu não acabo nunca. |
| 58 | você → **computador** | Para a música. |
| 59 | mãe → você | Tá bom, não esquece de descer pra almoçar. |
| 60 | você → mãe | Pode deixar, encosta a porta quando sair, por favor. |

## Cena 4 - De volta ao trabalho

*De novo sem ninguém no quarto, revisando o capítulo 3.*

| # | Quem → para quem | Fala |
|---|---|---|
| 61 | você → **computador** | Volta para o documento do capítulo três. |
| 62 | você → **computador** | Procura no texto onde aparece a palavra limiar. |
| 63 | você → (pensando alto) | Opa, aqui tá errado, não foi esse o limiar que a gente usou. |
| 64 | você → **computador** | Apaga esse parágrafo inteiro. |
| 65 | você → **computador** | Escreve o limiar foi escolhido na validação e o teste foi avaliado uma única vez. |
| 66 | você → **computador** | Lê em voz alta o último parágrafo. |
| 67 | você → (pensando alto) | Ficou bom, mas dá pra melhorar o final. |
| 68 | você → **computador** | Salva o documento. |
| 69 | você → **computador** | Converte o capítulo para PDF e salva na pasta do TCC. |
| 70 | você → (pensando alto) | Ufa, agora só falta a parte do sistema ao vivo. |
| 71 | você → **computador** | Tira um print dessa tela e salva na área de trabalho. |
| 72 | você → **computador** | Abre uma nova aba e pesquisa como citar um site nas normas da ABNT. |
| 73 | você → **computador** | Entra no primeiro resultado. |
| 74 | você → (pensando alto) | Esse site tá cheio de propaganda, que coisa chata. |
| 75 | você → **computador** | Volta para a página anterior. |
| 76 | você → **computador** | Diminui o brilho da tela, que tá machucando o olho. |
| 77 | você → (pensando alto) | Esse computador tá esquentando demais hoje. |
| 78 | você → **computador** | Fecha os programas que eu não tô usando. |

## Cena 5 - Seu irmão entra no quarto

*Seu irmão mais novo abre a porta procurando o carregador.*

| # | Quem → para quem | Fala |
|---|---|---|
| 79 | irmão → você | Ei, você viu o meu carregador? |
| 80 | você → irmão | Tá ali em cima da escrivaninha, do lado da impressora. |
| 81 | irmão → você | Valeu, e qual é mesmo a senha do Wi-Fi? |
| 82 | você → irmão | Pera, que eu vou ver aqui pra você. |
| 83 | você → **computador** | Mostra a senha da rede Wi-Fi. |
| 84 | você → irmão | Anota aí, tá aparecendo na tela. |
| 85 | irmão → você | Já conectou, você pode imprimir um trabalho da escola pra mim? |
| 86 | você → irmão | Posso, me manda o arquivo por e-mail. |
| 87 | irmão → você | Já mandei, é o último que chegou. |
| 88 | você → **computador** | Imprime o anexo do último e-mail. |
| 89 | você → (pensando alto) | Essa impressora sempre demora pra começar. |
| 90 | você → irmão | Ih, saiu só de um lado, a escola pediu frente e verso? |
| 91 | irmão → você | Pediu sim, frente e verso. |
| 92 | você → **computador** | Imprime de novo, só que frente e verso. |
| 93 | irmão → você | Agora sim, valeu, fico te devendo essa. |
| 94 | você → irmão | De nada, depois você me paga um sorvete. |

## Cena 6 - Ligação no celular

*A Fernanda liga. Só a sua parte é falada: deixe uns 3 s entre as linhas, como se ouvisse a resposta.*

| # | Quem → para quem | Fala |
|---|---|---|
| 95 | você → Fernanda | Oi, Fernanda, pode falar. |
| 96 | você → Fernanda | Não, eu ainda não recebi o seu e-mail. |
| 97 | você → Fernanda | Manda de novo pra mim, por favor, no endereço da faculdade. |
| 98 | você → Fernanda | Isso, e coloca o arquivo em PDF, que fica mais fácil de abrir. |
| 99 | você → Fernanda | Pera aí, deixa eu ver se chegou. |
| 100 | você → **computador** | Atualiza a caixa de entrada. |
| 101 | você → Fernanda | Chegou, tá aqui, valeu! |
| 102 | você → Fernanda | Tô abrindo aqui, só um segundo. |
| 103 | você → **computador** | Abre a apresentação que a Fernanda mandou. |
| 104 | você → Fernanda | Abre aí o slide sete e aumenta a fonte do gráfico, que tá muito pequena. |
| 105 | você → Fernanda | Beleza, depois a gente conversa, tchau! |

## Cena 7 - Pausa para o almoço

| # | Quem → para quem | Fala |
|---|---|---|
| 106 | você → **computador** | Cria um lembrete pra amanhã às nove da manhã pra eu mandar o capítulo pro meu orientador. |
| 107 | você → **computador** | Fecha todas as abas do navegador. |
| 108 | você → (pensando alto) | Chega por agora, minha cabeça já não funciona mais. |
| 109 | você → (pensando alto) | Depois do almoço eu volto pra terminar a conclusão. |
| 110 | você → **computador** | Coloca no modo não perturbe. |
| 111 | mãe → você | Desce que o almoço tá na mesa! |
| 112 | você → mãe | Tô indo! |
| 113 | você → **computador** | Bloqueia a tela que eu vou almoçar. |

## Resumo

- 113 falas: 42 dirigidas ao computador (DD) e 71 não dirigidas (NDD), em 7 cenas.
- Duração estimada: 11 a 13 min (901 palavras a 2,0-2,6 palavras por segundo,
  mais as pausas entre as linhas e entre as cenas).
- Uma das falas dirigidas é da mãe (55): qualquer pessoa da sala pode falar com o computador.
- Casos difíceis de propósito:
  - pedidos a pessoas com vocabulário de computador (23, 28, 32, 36, 49, 84, 85, 86, 97, 98, 104);
  - o mesmo pedido feito primeiro a uma pessoa e depois ao computador (28 → 29, 36 → 41, 53 → 54);
  - comentários sobre o computador ou a impressora (52, 74, 77, 89);
  - falas de até 4 palavras nas duas classes: DD (22, 58, 64, 68, 73) e NDD
    (16, 35, 38, 39, 45, 50, 95, 101, 112).
- Nenhuma fala é igual ou quase igual (cosseno TF-IDF > 0,95, como na validação externa) a uma frase dos
  corpora com e sem correção ou do roteiro de conversa das gravações (`config/roteiro_conversa.txt`).
