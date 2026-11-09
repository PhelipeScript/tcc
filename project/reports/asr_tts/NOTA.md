# Nota sobre esta execução (2026-11-04, 21h36-21h43)

- **Whisper:** large-v3-turbo no mlx, com a mesma configuração do sistema e a semente fixa
  (`[asr] seed = 0`).
- **Energia:** na tomada, com 100%, sem Modo de Pouca Energia.
- **Log:** sem avisos nem erros (`data/logs/eval_asr_tts_semente.log`).
- **Dados:** 740 frases do teste interno (370 DD e 370 NDD), estratificadas por classe e faixa de tamanho e
  sintetizadas com vozes pt-BR do edge-tts. Cada frase aparece uma vez.
- **Método:** o mesmo item é classificado no texto lido e na transcrição do Whisper (pareado). O bootstrap
  sorteia os itens.

## Resultado

- **WER médio:** 0,049. Em 580 das 740 frases, a transcrição saiu sem erro.
- **Acerto na transcrição:** Albertina 98,2%, BERTimbau e DeBERTinha 97,4%, TF-IDF 94,9% (os dois).
- **Hipótese nas transcrições:** em `hipotese.md`.
- **Queda do acerto por faixa de WER:** em `hipotese.md`. A comparação da queda entre os modelos está em
  `robustez.md`.
- **Índice de confiança × WER:** Spearman de -0,029 (p = 0,42). Na voz sintética quase não há erro de ASR
  para a confiança acompanhar. Na voz do autor, o valor é -0,291 (`asr_gravacoes/NOTA.md`).
