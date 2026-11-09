# Sistema completo - gravacoes (192 itens), bertimbau

ativação = fração de itens em que o sistema decidiu DD (em DD é o acerto; em NDD e RUÍDO, falso positivo). 'política baixa→NDD' = como ficaria se confiabilidade baixa forçasse NDD.

| grupo | rótulo | n | ativação | ativação (política) | sem fala | falas/item | latência p50 (s) | latência p95 (s) |
|---|---|---|---|---|---|---|---|---|
| conversa_interno | NDD | 24 | 0.0000 | 0.0000 | 0.0000 | 1.0417 | 1.1721 | 1.2174 |
| distante_interno | DD | 20 | 0.9500 | 0.9500 | 0.0000 | 1.0000 | 1.1602 | 1.3442 |
| distante_interno | NDD | 20 | 0.1500 | 0.1500 | 0.0000 | 1.1500 | 1.1602 | 1.3442 |
| distante_interno | NDD_LIVRE | 1 | 0.0000 | 0.0000 | 0.0000 | 2.0000 | 1.1602 | 1.3442 |
| distante_interno | RUIDO | 1 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.1602 | 1.3442 |
| quieto_interno | DD | 20 | 0.9500 | 0.9500 | 0.0000 | 1.0000 | 1.2119 | 1.3611 |
| quieto_interno | NDD | 20 | 0.1000 | 0.1000 | 0.0000 | 1.0000 | 1.2119 | 1.3611 |
| quieto_interno | NDD_LIVRE | 1 | 0.0000 | 0.0000 | 0.0000 | 3.0000 | 1.2119 | 1.3611 |
| quieto_interno | RUIDO | 1 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.2119 | 1.3611 |
| quieto_iphone | DD | 20 | 0.9500 | 0.9500 | 0.0000 | 1.0000 | 1.1843 | 1.2838 |
| quieto_iphone | NDD | 20 | 0.1000 | 0.1000 | 0.0000 | 1.0000 | 1.1843 | 1.2838 |
| quieto_iphone | NDD_LIVRE | 1 | 0.0000 | 0.0000 | 0.0000 | 3.0000 | 1.1843 | 1.2838 |
| quieto_iphone | RUIDO | 1 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.1843 | 1.2838 |
| ruido_interno | DD | 20 | 0.9500 | 0.9500 | 0.0000 | 1.0000 | 1.1681 | 1.3523 |
| ruido_interno | NDD | 20 | 0.0500 | 0.0500 | 0.0000 | 1.0000 | 1.1681 | 1.3523 |
| ruido_interno | NDD_LIVRE | 1 | 0.0000 | 0.0000 | 0.0000 | 2.0000 | 1.1681 | 1.3523 |
| ruido_interno | RUIDO | 1 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.1681 | 1.3523 |

### Fala livre (NDD sem roteiro): falsos positivos por fala detectada

Cada arquivo tem várias falas, então a ativação por arquivo da tabela acima não se compara à das frases. Aqui o FP é contado por fala e por minuto de áudio.

| grupo | arquivos | minutos | falas | FP por fala | FP por fala (política) | ativações/min |
|---|---|---|---|---|---|---|
| distante_interno | 1 | 0.7000 | 2 | 0.0000 | 0.0000 | 0.0000 |
| quieto_interno | 1 | 0.6000 | 3 | 0.0000 | 0.0000 | 0.0000 |
| quieto_iphone | 1 | 1.7000 | 3 | 0.0000 | 0.0000 | 0.0000 |
| ruido_interno | 1 | 0.6000 | 2 | 0.0000 | 0.0000 | 0.0000 |
