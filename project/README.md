# DDSD textual em PT-BR

Um assistente de voz sem palavra de ativação precisa decidir, a cada fala, se ela é dirigida ao computador
(DD) ou não (NDD). Este projeto toma essa decisão só pelo texto: o Whisper transcreve a fala e um
classificador decide a partir da transcrição.

Código do trabalho de conclusão de curso "Detecção de Fala Direcionada ao Computador Pessoal em Português
Brasileiro: Uma Abordagem Textual para Assistentes de Voz sem Palavra de Ativação", de Phelipe Pereira de
Souza (SENAC Santo Amaro, 2026).

## Conteúdo

| Pasta | O que tem |
|---|---|
| `ddsd/` | O pacote, com a CLI `ddsd`: corpus, treino, avaliação e o sistema em tempo real |
| `config/` | Configuração do corpus (`corpus.toml` e `corpus_controle.toml`) e do sistema (`pipeline.toml`) |
| `docs/` | Arquitetura do sistema, protocolo experimental, critério de anotação e roteiro da sessão ao vivo |
| `reports/` | Resultados: relatórios em Markdown e JSON, figuras em `reports/figuras/` e tabelas em `reports/tabelas_tcc.md` |
| `scripts/` | Figuras, tabelas e relatórios do treino e do corpus; treino completo e sistema com os 5 modelos |
| `tests/` | Testes unitários e de integração |

`data/` e `models/` ficam fora do git.

## Sistema em tempo real

Feito para macOS com Apple Silicon: o Whisper (large-v3-turbo) roda na GPU pelo mlx. Precisa do
[uv](https://docs.astral.sh/uv/) e, no terminal, de permissão de microfone.

```bash
uv sync
uv run ddsd download-models   # Whisper, Silero VAD e os classificadores do Hugging Face
uv run ddsd devices           # lista as entradas de áudio
uv run ddsd run               # microfone, com painel no terminal
```

- Os classificadores ficam em `models/` (cerca de 1,1 GB). Se faltar algum, o sistema o baixa do Hugging Face
  ao carregá-lo.
- O classificador padrão é o BERTimbau. `--classificador` escolhe outro (`albertina`, `debertinha`,
  `baseline_logreg` ou `baseline_svm`); no painel, as teclas 1, 2 e 3 trocam entre os Transformers.
- `--backend faster-whisper` roda o Whisper na CPU; `--no-dashboard` troca o painel por linhas de log.
- `--gravar` guarda o áudio de cada fala e as evidências de cada decisão em `data/sessoes/` (grava voz).
- `uv run ddsd replay fala.wav` passa gravações pelo mesmo pipeline.

A arquitetura (captura, detector de voz, Whisper, classificador e decisão) está em `docs/arquitetura.md`.

## Modelos e corpus no Hugging Face

| Repositório | Conteúdo |
|---|---|
| `phelipescript/ddsd-ptbr-bertimbau` | BERTimbau, o classificador padrão |
| `phelipescript/ddsd-ptbr-albertina` | Albertina |
| `phelipescript/ddsd-ptbr-debertinha` | DeBERTinha |
| `phelipescript/ddsd-ptbr-tfidf` | TF-IDF + Regressão Logística e TF-IDF + SVM |
| `phelipescript/ddsd-ptbr-<modelo>-sem-correcao` | Os 3 Transformers treinados no corpus sem a correção do viés de comprimento |
| `phelipescript/ddsd-ptbr-corpus` (dataset) | O corpus, com as partições de treino, validação e teste, e as fontes em `fontes/` |

Cada modelo traz o limiar de decisão escolhido na validação.

```python
from datasets import load_dataset

corpus = load_dataset("phelipescript/ddsd-ptbr-corpus")  # com correção (padrão)
controle = load_dataset("phelipescript/ddsd-ptbr-corpus", "sem_correcao")
```

## Reproduzir o experimento

O procedimento, com o comando de cada etapa, está em `docs/protocolo_experimental.md`.

1. **Corpus.** As fontes publicadas reconstroem as mesmas partições dos dois corpora:
   ```bash
   uv run hf download phelipescript/ddsd-ptbr-corpus --repo-type dataset --include "fontes/*" --local-dir data/corpus/hf
   mv data/corpus/hf/fontes data/corpus/sources
   uv run ddsd corpus build
   uv run ddsd corpus freeze
   uv run ddsd corpus build --config config/corpus_controle.toml
   uv run ddsd corpus freeze --config config/corpus_controle.toml
   ```
2. **Treino** (MPS): `scripts/treino_completo.sh` treina as duas condições e retoma o que parou. Equivale a
   `uv run ddsd train all --seeds 42 43 44` e, para os modelos sem correção,
   `uv run ddsd train all --seeds 42 --data-dir data/corpus/control/frozen/splits --output-dir models/controle`.
3. **Avaliação:** `uv run ddsd eval <ação>` e `uv run ddsd benchmark`. `scripts/sessao_pipeline.sh` roda o
   sistema completo com os 5 modelos num conjunto de áudio. A seção 3 do protocolo liga cada experimento ao
   comando e ao relatório em `reports/`.
4. **Figuras e tabelas:** `uv run python scripts/figuras.py` regera `reports/figuras/`, e
   `uv run python scripts/tabelas_tcc.py` regera `reports/tabelas_tcc.md`, a partir dos relatórios.

A geração de frases com LLMs (`ddsd generate`) não é necessária para reproduzir o corpus. Ela usa as chaves do
`.env` (modelo em `.env.example`).

## Testes

```bash
uv run pytest -m "not models and not audio_hw"   # sem modelos nem hardware de áudio
uv run pytest                                    # tudo
```

- Os testes `models` usam o Whisper do cache do Hugging Face (`uv run ddsd download-models`).
- Os testes `audio_hw` usam o microfone e tocam uma frase no alto-falante.
- O teste de ponta a ponta do pipeline usa os classificadores do teste de fumaça e é pulado sem eles. Eles são
  treinados com o corpus já construído (passo 1 acima):

  ```bash
  uv run ddsd train bertimbau --smoke
  uv run ddsd train debertinha --smoke
  uv run ddsd train albertina --smoke --precision fp32
  ```

## Limitações

As limitações conhecidas estão na seção 4 de `docs/protocolo_experimental.md`. Entre elas: o corpus é 100%
sintético, as gravações feitas para o trabalho têm um único falante (o autor) e o sistema não separa
falantes.
