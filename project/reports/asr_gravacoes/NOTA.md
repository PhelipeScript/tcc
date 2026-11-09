# Nota sobre esta execução (2026-11-04, 18h48-18h50)

- **Whisper:** large-v3-turbo no mlx, com a mesma configuração do sistema e a semente fixa
  (`[asr] seed = 0`).
- **Energia:** na tomada, sem Modo de Pouca Energia.
- **Log:** sem avisos nem erros (`data/logs/eval_asr_gravacoes_semente.log`).
- **Dados:** a voz do autor, um único falante, gravada a 48 kHz em 2026-11-04.
  - **Frases:** as mesmas 40 frases do teste (20 DD e 20 NDD) em 4 cenários: sala quieta, ruído de fundo,
    iPhone (Continuity) e ~2 m do computador.
  - **Roteiro de conversa:** 24 falas NDD (`config/roteiro_conversa.txt`), lidas como numa conversa com
    uma pessoa na sala.
  - **Fala livre:** ~40 s por cenário, sem roteiro, por isso fora do WER.
- **Método:** o mesmo item é classificado no texto lido e na transcrição do Whisper (pareado).

## Resultado (BERTimbau)

| Conjunto | Itens | WER | Acerto no texto lido | Acerto na transcrição | Falsas ativações | Pedidos perdidos |
|---|---|---|---|---|---|---|
| Sala quieta | 40 | 0,137 | 95,0% | 92,5% | 2/20 | 1/20 |
| Ruído de fundo | 40 | 0,220 | 95,0% | 95,0% | 1/20 | 1/20 |
| iPhone | 40 | 0,205 | 95,0% | 90,0% | 2/20 | 2/20 |
| ~2 m | 40 | 0,201 | 95,0% | 92,5% | 2/20 | 1/20 |
| Roteiro de conversa | 24 | 0,031 | 100% | 100% | 0/24 | - |

- **WER geral:** 0,170, nos 184 itens com texto de referência.
- **Fala livre:** 0 ativações nos 4 trechos.
- **Índice de confiança × WER:** Spearman de -0,291 (p = 6 × 10⁻⁵). Na fala real, a confiança acompanha os
  erros do ASR. O p trata as 4 leituras de cada frase como independentes, então é otimista.
- **WER inflado nas frases curtas:** um erro numa frase de 2 palavras já vale 50%. Também contam as
  variações de grafia, como "compila" → "compilar" e "trocar tema" → "trocar o tema".
- **Margem:** com 40 frases por cenário, cada erro vale 2,5 p.p. As diferenças entre cenários estão dentro
  dessa margem.
- **Comparações:**
  - entre os modelos, em `resumo.md` (todos);
  - a hipótese nas transcrições, em `hipotese.md`;
  - a queda do acerto, em `robustez.md`.
  - Nas duas últimas, o bootstrap sorteia as frases, e as leituras de cada frase nos 4 cenários entram juntas.
    - Na acurácia, na F1 e no EER, nenhum par de modelos difere de forma significativa na voz do autor.
    - Na ROC-AUC, o BERTimbau supera os dois TF-IDF: +4,0 p.p. [+0,7; +8,7] contra a Regressão Logística e
      +3,6 p.p. [+0,2; +8,5] contra o SVM.
