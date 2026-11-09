# Nota sobre estas execuções (2026-11-04, 21h46, a 2026-11-05, 03h22)

- **O que é:** o sistema completo em tempo real, com os 5 modelos com correção, cada um na sua execução e no
  seu processo. Os 3 modelos sem correção julgaram as transcrições da execução do BERTimbau.
- **Dados:** a voz sintética, as mesmas 740 frases de `asr_tts/` (370 DD e 370 NDD do teste interno, em 3 vozes
  pt-BR do edge-tts), tocadas em tempo real no buffer do microfone, com 1,5 s de silêncio depois de cada uma.
  - As frases são do mesmo tipo das de treino, e a voz é limpa: é o melhor caso do sistema ao vivo.
- **Mesmo áudio e mesma transcrição para todos:** semente fixa do Whisper (`[asr] seed = 0`), a mesma ordem
  dos itens e um processo por modelo.
  - Nas 5 execuções, 740 de 740 itens tiveram os mesmos trechos e a mesma transcrição.
  - Paridade: o BERTimbau, julgando de novo as transcrições, reproduziu 751 de 751 decisões.
  - O detector de voz fez uma fala de 729 itens e duas de 11 (751 falas).
- **Código:** o mesmo nas 5 execuções e na comparação.
- **Energia:** na tomada, com 100%, do início ao fim das 5 execuções. Os logs registram isso no início e no
  fim de cada execução, e o `pmset -g log` não tem nenhuma passagem para a bateria no período.
- **Logs** (`data/logs/pipeline_tts_<modelo>.log`): sem erros. Cada execução teve de 390 a 392 avisos, todos
  de "entrada digitalmente silenciosa".
  - Os arquivos do edge-tts começam e terminam com silêncio digital (amostras exatamente zero): mediana de
    0,05 s no início e 0,41 s no fim.
  - Somado ao 1,5 s de pausa entre os itens, esse silêncio passa dos 2 s do alerta de microfone mudo
    (`silent_input_s`).
  - O aviso só vai para o log e não muda nenhuma decisão.

## Resultado

| Modelo | Pedidos atendidos (DD) | Falsas ativações (NDD) |
|---|---|---|
| BERTimbau | 357/370 (96,5%) | 7/370 (1,9%) |
| TF-IDF + Regressão Logística | 345/370 (93,2%) | 15/370 (4,1%) |
| TF-IDF + SVM | 349/370 (94,3%) | 19/370 (5,1%) |
| DeBERTinha | 355/370 (95,9%) | 7/370 (1,9%) |
| Albertina | 358/370 (96,8%) | 4/370 (1,1%) |
| BERTimbau sem correção | 348/370 (94,1%) | 127/370 (34,3%) |
| DeBERTinha sem correção | 348/370 (94,1%) | 134/370 (36,2%) |
| Albertina sem correção | 348/370 (94,1%) | 121/370 (32,7%) |

- **Por voz:** a tabela completa está em `comparacao.md`.
- **Comparado ao Whisper offline** no arquivo inteiro (`asr_tts/`): o sistema ao vivo erra de 1 a 3 itens a
  mais por modelo.

## Diferenças pareadas (bootstrap por item)

Cada frase aparece uma vez, então o bootstrap sorteia os itens.

| Comparação | Diferença | IC 95% |
|---|---|---|
| Sem correção × com correção, falsas ativações: BERTimbau | +32,4 p.p. | [+27,6; +37,3] |
| Sem correção × com correção, falsas ativações: DeBERTinha | +34,3 p.p. | [+29,5; +39,5] |
| Sem correção × com correção, falsas ativações: Albertina | +31,6 p.p. | [+26,8; +36,5] |
| TF-IDF + Regressão Logística × BERTimbau, pedidos atendidos | -3,2 p.p. | [-5,9; -0,5] |
| TF-IDF + SVM × BERTimbau, falsas ativações | +3,2 p.p. | [+0,8; +5,9] |
| TF-IDF + Regressão Logística × BERTimbau, falsas ativações | +2,2 p.p. | [0,0; +4,6] |
| TF-IDF + SVM × BERTimbau, pedidos atendidos | -2,2 p.p. | [-4,6; +0,3] |
| DeBERTinha e Albertina × BERTimbau | de -0,8 a +0,3 p.p. | incluem o zero |

- **Sem correção:** os pedidos atendidos caem de 1,9 a 2,7 p.p., sem significância. O IC da Albertina vai
  até 0,0.
- **Regressão Logística, falsas ativações:** o IC começa em 0,0, então a diferença não é significativa.

## De onde vêm os erros

| Modelo | DD: segmentação / classificador / transcrição | NDD: segmentação / classificador / transcrição |
|---|---|---|
| BERTimbau | 1 / 9 / 3 | 1 / 3 / 3 |
| TF-IDF + Regressão Logística | 0 / 16 / 9 | 2 / 11 / 2 |
| TF-IDF + SVM | 0 / 12 / 9 | 2 / 15 / 2 |
| DeBERTinha | 1 / 11 / 3 | 0 / 6 / 1 |
| Albertina | 0 / 8 / 4 | 0 / 4 / 0 |
| BERTimbau sem correção | 0 / 21 / 1 | 3 / 123 / 1 |
| DeBERTinha sem correção | 0 / 20 / 2 | 3 / 130 / 1 |
| Albertina sem correção | 0 / 20 / 2 | 2 / 119 / 0 |

- **As três origens:** definidas em `pipeline_gravacoes/NOTA.md`.
- **Classificador:** é a maior parte dos erros, porque o Whisper quase não erra nessa voz (WER de 4,9%).
- **Transcrição:** os TF-IDF perdem 9 pedidos por erro de transcrição, e os Transformers, de 3 a 4.

## Latência (do fim da fala até a decisão, com os 0,7 s de espera do detector de voz)

| Execução | Whisper p50 | Total p50 | Total p95 | Classificador p50 |
|---|---|---|---|---|
| BERTimbau | 0,42 s | 1,17 s | 1,27 s | 38 ms |
| TF-IDF + Regressão Logística | 0,42 s | 1,13 s | 1,20 s | 1,4 ms |
| TF-IDF + SVM | 0,41 s | 1,12 s | 1,20 s | 4,3 ms |
| DeBERTinha | 0,42 s | 1,16 s | 1,25 s | 31 ms |
| Albertina | 0,41 s | 1,16 s | 1,24 s | 43 ms |

- **751 falas por execução.** Nenhuma precisou de nova tentativa do Whisper.
- **Whisper:** o mesmo em todas, com o mesmo áudio: mediana de 0,41 a 0,42 s.
- **Classificador:** de 1,4 a 43 ms, até 4% do total.
