# Nota sobre estas execuções (2026-11-05, 09h21-14h18)

- **O que é:** o sistema completo em tempo real, com os 5 modelos com correção, cada um na sua execução e no
  seu processo. Os 3 modelos sem correção julgaram as transcrições da execução do BERTimbau.
- **Dados:** 600 trechos de conversa espontânea, todos NDD: 300 do CORAA e 300 do NURC-SP.
  - São trechos de 1,5 a 15 s das partições de teste, sorteados com a semente 42 (`ddsd eval audio-prepare`).
  - Foram tocados em tempo real no buffer do microfone, com 1,5 s de silêncio depois de cada um.
  - Como só há NDD, este conjunto mede as falsas ativações. Os pedidos atendidos são medidos na voz do autor e
    na voz sintética.
- **Mesmo áudio e mesma transcrição para todos:** semente fixa do Whisper (`[asr] seed = 0`), a mesma ordem
  dos itens e um processo por modelo.
  - Nas 5 execuções, 600 de 600 itens tiveram os mesmos trechos e a mesma transcrição. Isso inclui as 39 falas
    em que o Whisper fez uma nova tentativa, que é a parte sorteada da transcrição.
  - Paridade: o BERTimbau, julgando de novo as transcrições, reproduziu 691 de 691 decisões.
- **Falas:** o detector de voz fez 691 falas, todas fechadas pelo fim da voz.
  - 519 itens viraram uma fala, e 77 viraram de duas a quatro.
  - Em 4 itens, nenhuma fala foi detectada: `coraa_0237`, `coraa_0268`, `nurcsp_0066` e `nurcsp_0274`.
  - Em 2 falas, o Whisper não devolveu texto, e a decisão foi NDD (`coraa_0178` e `coraa_0233`).
- **Código:** o mesmo nas 5 execuções e na comparação. Em relação à voz sintética, só mudaram
  os comandos de relatório (`asr-relatorio` e `asr-hipotese`).
- **Energia:** na tomada, com 100%, do início ao fim. Os logs registram isso no início e no fim de cada
  execução, e o `pmset -g log` não tem nenhuma passagem para a bateria nem repouso no período.
- **Logs:** sem avisos nem erros (`data/logs/pipeline_publico_<modelo>.log` e `data/logs/comparar_publico.log`).

## Resultado

| Modelo | Falsas ativações | CORAA | NURC-SP |
|---|---|---|---|
| BERTimbau | 39/600 (6,5%) | 22/300 (7,3%) | 17/300 (5,7%) |
| TF-IDF + Regressão Logística | 51/600 (8,5%) | 29/300 (9,7%) | 22/300 (7,3%) |
| TF-IDF + SVM | 52/600 (8,7%) | 31/300 (10,3%) | 21/300 (7,0%) |
| DeBERTinha | 32/600 (5,3%) | 15/300 (5,0%) | 17/300 (5,7%) |
| Albertina | 31/600 (5,2%) | 14/300 (4,7%) | 17/300 (5,7%) |
| BERTimbau sem correção | 175/600 (29,2%) | 104/300 (34,7%) | 71/300 (23,7%) |
| DeBERTinha sem correção | 194/600 (32,3%) | 123/300 (41,0%) | 71/300 (23,7%) |
| Albertina sem correção | 142/600 (23,7%) | 88/300 (29,3%) | 54/300 (18,0%) |

- **Índice de confiança:** se a confiança baixa forçasse NDD, as falsas ativações cairiam para 3,7-4,2% nos
  Transformers e 7,3-7,5% nos TF-IDF. Essa política só é avaliada offline: no sistema, a confiança baixa
  gera um alerta e não muda a decisão.

## Diferenças pareadas (bootstrap por item)

Cada trecho aparece uma vez, então o bootstrap sorteia os itens.

| Comparação | Diferença | IC 95% |
|---|---|---|
| Sem correção × com correção: BERTimbau | +22,7 p.p. | [+19,2; +26,2] |
| Sem correção × com correção: DeBERTinha | +27,0 p.p. | [+23,3; +30,7] |
| Sem correção × com correção: Albertina | +18,5 p.p. | [+15,3; +21,8] |
| TF-IDF + Regressão Logística × BERTimbau | +2,0 p.p. | [-0,5; +4,5] |
| TF-IDF + SVM × BERTimbau | +2,2 p.p. | [-0,3; +4,8] |
| DeBERTinha × BERTimbau | -1,2 p.p. | [-3,0; +0,7] |
| Albertina × BERTimbau | -1,3 p.p. | [-3,3; +0,7] |

- **Correção:** é a única diferença significativa.
- **Entre os modelos com correção:** nenhuma diferença contra o BERTimbau é significativa.
- **De onde vêm os erros:** a separação entre classificador e transcrição usa o impacto do ASR
  (`asr_<conjunto>/`), que não foi medido neste conjunto.

## Latência (do fim da fala até a decisão, com os 0,7 s de espera do detector de voz)

| Execução | Whisper p50 | Total p50 | Total p95 | Classificador p50 |
|---|---|---|---|---|
| BERTimbau | 0,42 s | 1,15 s | 1,91 s | 28 ms |
| TF-IDF + Regressão Logística | 0,42 s | 1,12 s | 1,85 s | 0,6 ms |
| TF-IDF + SVM | 0,42 s | 1,13 s | 1,86 s | 4,3 ms |
| DeBERTinha | 0,42 s | 1,16 s | 1,85 s | 31 ms |
| Albertina | 0,42 s | 1,18 s | 1,90 s | 44 ms |

- **691 falas por execução.**
- **Nova tentativa do Whisper:** 39 das 691 falas (21 do CORAA e 18 do NURC-SP). Elas formam a cauda da
  latência:
  - as 35 falas mais lentas (o topo de 5%) são todas desse tipo;
  - a mediana delas é de 3,1 s, contra 1,1-1,2 s nas demais;
  - sem elas, o p95 fica entre 1,23 e 1,28 s.
- **Classificador:** de 0,6 a 44 ms, até 4% do total. O do BERTimbau ficou em 28 ms aqui e em 38-41 ms nos
  outros conjuntos, sem causa conhecida; DeBERTinha e Albertina ficaram iguais nos três.
