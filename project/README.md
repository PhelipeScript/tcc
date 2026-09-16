# Documentação técnica — `project/`

Este documento cobre **como instalar, rodar e entender o código**. Para o andamento do
trabalho (o que já foi feito, resultados, cronograma), veja o [README raiz](../README.md).

## Instalação

Gerenciado via [`uv`](https://docs.astral.sh/uv/). Requer Python 3.11+.

```bash
cd project
uv sync
```

`torch` é resolvido por plataforma (CUDA no Windows/Linux, CPU no macOS) — nenhuma flag
extra é necessária. Chaves de API para os geradores de dataset vão em `.env`
(`.env.example` documenta as variáveis esperadas).

## Estrutura

```
project/
├── data/                    # gerado por src/data_prep; não versionado
│   ├── 01_snapshot/ ... 07_splits/   # uma pasta por estágio do pipeline
│   └── reports/             # pipeline_report.{md,json} — estatísticas de cada execução
├── outputs/                 # gerado por src/training; não versionado
│   └── <modelo>/            # checkpoints, métricas, gráficos, manifest por modelo
└── src/
    ├── dataset/             # Etapa 1 — geração sintética via LLMs
    ├── data_prep/           # Etapa 2 — pipeline de preparação (7 estágios)
    └── training/            # Etapa 3 — fine-tuning e avaliação
```

## Etapa 1 — Geração do dataset (`src/dataset/`)

Um script por provedor (`{glm,groq,lmstudio,mistral,nvidia,openrouter,qwen}_generate.py`),
todos seguindo o mesmo contrato: leem a taxonomia compartilhada em
`src/dataset/prompts/context_schema.yaml` (7 categorias × 8 contextos × 6 tons), geram
por 3 estratégias de prompt (paráfrase, roleplay com persona, contraste NDD), filtram
*wake-words* e vocabulário de casa inteligente, e gravam incrementalmente em
`src/dataset/<fonte>/{dd,ndd}.jsonl` via `tenacity` (retry em falha de API).

```bash
uv run python -m src.dataset.<fonte>_generate
```

**Nunca rodar duas instâncias do mesmo script em paralelo** — a escrita incremental no
mesmo arquivo `.jsonl` corrompe o resultado. `opencode` e `sonnet` não têm script: são
gerados em sessões interativas de CLI (OpenCode / Claude Sonnet).

## Etapa 2 — Preparação do dataset (`src/data_prep/`)

Pipeline de 7 estágios, cada um um módulo isolado que lê a saída do anterior. As pastas
de origem (`src/dataset/<fonte>/`) são tratadas como somente-leitura.

```bash
uv run python -m src.data_prep.run_pipeline
```

| Estágio | Módulo | Função |
|---|---|---|
| 1 | `stage1_snapshot.py` | Copia as fontes, grava manifesto (contagem, mtime, SHA-256) |
| 2 | `stage2_normalize.py` | NFKD → remove acento → minúsculas → remove pontuação → colapsa espaço |
| 3 | `stage3_dedup_per_file.py` | Remove duplicatas dentro de cada arquivo |
| 4 | `stage4_merge.py` | Consolida as fontes em `dd.jsonl`/`ndd.jsonl` |
| 5 | `stage5_dedup_final.py` | Dedup entre fontes + remove conflitos de rótulo (`conflicts.jsonl`) |
| 6 | `stage6_balance.py` | Undersample da classe majoritária até 1:1 (`discarded.jsonl`) |
| 7 | `stage7_split.py` | Split 80/10/10 estratificado por `label + source`, sem overlap de texto |

Configuração central em `config.py` (`PROVIDERS`, `MIN_CHARS`, `SPLIT_RATIOS`, `SEED=8`).
Cada execução grava `data/reports/pipeline_report.{md,json}` com as contagens de cada
estágio — é a fonte de verdade para qualquer número citado no texto do TCC sobre o
dataset.

Schema de saída (`data/07_splits/{train,val,test}.jsonl`):

```json
{"text": "fecha essa aba por favor", "text_raw": "Fecha essa aba, por favor!", "label": "DD", "source": "glm"}
```

`text` é a entrada usada no treino (normalizada); `text_raw` existe para permitir a
ablação com acentuação/pontuação preservadas sem reprocessar nada.

## Etapa 3 — Treinamento (`src/training/`)

```bash
uv run python -m src.training.train_all              # os 3 modelos em sequência
uv run python -m src.training.train_bertimbau         # um modelo isolado
uv run python -m src.training.train_bertimbau --epochs 2 --allow-cpu --max-steps 50  # smoke test
```

Os três scripts de treino (`train_bertimbau.py`, `train_albertina.py`,
`train_debertinha.py`) são wrappers finos que chamam `cli.main_for_model(<chave>)`; todas
as opções vêm de `config.py` (`ModelConfig` por modelo, `TrainingConfig` comum) e podem
ser sobrescritas via linha de comando (`--help` lista todas).

**Ordem de execução fixa em `engine.py`** (não alterar sem motivo forte — é o que sustenta
a validade metodológica dos resultados):

```
hardware → semente → dados → modelo → treino
→ avalia VALIDAÇÃO → escolhe limiar de decisão na VALIDAÇÃO
→ avalia TESTE uma única vez, com o limiar congelado
→ grava artefatos
```

Cada execução grava em `outputs/<modelo>/`:

| Arquivo | Conteúdo |
|---|---|
| `checkpoints/checkpoint-<step>/` | Pesos, otimizador, `trainer_state.json` (histórico de loss/métrica por época) |
| `metrics_val.json`, `metrics_test.json` | accuracy, precision/recall/F1 (DD e macro), ROC-AUC, PR-AUC, MCC, EER |
| `metrics_by_source_test.json` | As mesmas métricas, desagregadas por fonte geradora |
| `confusion_matrix_test.{csv,png}`, `roc_curve_test.png`, `det_curve_test.png` | Gráficos e tabela do conjunto de teste |
| `predictions_test.jsonl`, `errors_test.json` | Predição por exemplo e os casos errados |
| `run_manifest.json` | Hiperparâmetros, métricas, hardware, versões, SHA do commit |

`train_all.py` roda os três (da menor para a maior arquitetura) e gera
`outputs/comparison.{md,tex,json}` com a tabela comparativa.

`metrics.py` implementa EER por interpolação da curva ROC + `scipy.optimize.brentq`,
validado contra o caso analítico `d′=2` (EER teórico 0,1587, obtido 0,1590).
`reporting.py` gera todos os gráficos só com `matplotlib` (sem wandb/tensorboard — o
`Trainer` roda com `report_to=[]`).

### Ferramentas de inspeção manual

```bash
uv run python -m src.training.predict_interactive
```

CLI interativo: escolhe um dos 3 modelos, carrega o melhor checkpoint e o limiar de
decisão congelados a partir do `run_manifest.json`, e permite testar frases digitadas à
mão, acumulando métricas de sessão (`metricas` no menu).

```bash
uv run python -m src.training.podcast_eval "<url-do-youtube>" --model debertinha --secs 30
```

Baixa o áudio via `yt-dlp`, transcreve com `faster-whisper` (VAD ligado), agrupa em
blocos por segundos de fala, roda o classificador em cada bloco e reporta uma "timeline
de ativações" — quanto do áudio faria o assistente disparar. Salva `transcript.jsonl`,
`chunks.jsonl` e `summary.json` em `outputs/podcast_analysis/<id>_<slug>/`. É uma
validação qualitativa em fala real, não um substituto da validação externa formal
(Seção "Limitação conhecida" do README raiz).

## Baselines clássicos (`src/baseline/`)

```bash
uv run python -m src.baseline.train_baselines              # SVM + Regressão Logística
uv run python -m src.baseline.train_baselines --only svm
uv run python -m src.baseline.build_full_comparison         # tabela com os 3 Transformers + os 2 baselines
```

TF-IDF (n-gramas 1-3, vocabulário limitado a `max_features`) + `LinearSVC` calibrado
(`CalibratedClassifierCV`, já que `LinearSVC` não expõe `predict_proba`) e Regressão
Logística, ambos com `C` escolhido por `GridSearchCV` (5-fold estratificado) sobre o
treino. Segue a mesma disciplina do treino dos Transformers: limiar de decisão calibrado
só na validação (`src.training.metrics.best_threshold`), teste avaliado uma única vez.
Reaproveita `src.training.metrics` e `src.training.reporting` sem alterá-los — os
artefatos gravados em `outputs/baseline_svm/` e `outputs/baseline_logreg/` têm o mesmo
layout dos modelos Transformer.

`build_full_comparison.py` não treina nada: só lê `metrics_test.json` dos 5 modelos já
treinados e grava `outputs/comparison_full.{md,tex,json}`.

## Análise de overfitting e significância (`src/analysis/`)

```bash
uv run python -m src.analysis.run_analysis
```

Não treina nada — lê os artefatos que `src.training` e `src.baseline` já gravaram em
`outputs/` e gera, em `outputs/analysis/`:

| Subpasta | Conteúdo |
|---|---|
| `loss_curves/` | train_loss vs eval_loss por época (extraído do `trainer_state.json` do checkpoint de **maior step**, não o "melhor" — é o único que tem o histórico completo) + overlay dos 3 Transformers |
| `per_source/` | Gráfico de barras agrupadas de uma métrica por fonte geradora, para todos os modelos treinados |
| `calibration/` | Diagrama de confiabilidade (reliability diagram) sobre `predictions_test.jsonl`, ilustrando o overfitting de confiança |
| `bootstrap/` | Bootstrap pareado (2.000 reamostragens) entre todos os pares de modelos, com IC 95% e `prob_a_maior_que_b` por métrica |

O bootstrap pareado (`bootstrap.py`) reamostra o mesmo conjunto de índices do teste para
os dois modelos comparados a cada iteração — isso só é válido se as duas listas de
`predictions_test.jsonl` estiverem na mesma ordem de exemplos. `check_alignment()` valida
isso automaticamente (compara rótulo linha a linha) e aborta com erro claro se detectar
desalinhamento, em vez de produzir um IC silenciosamente inválido.

## ASR simulado via TTS → Whisper (`src/asr_eval/`)

```bash
uv run python -m src.asr_eval.run_asr_eval                                  # 750 exemplos, N=1,3,5
uv run python -m src.asr_eval.run_asr_eval --sample-size 100 --n-values 1 3  # execução menor
uv run python -m src.asr_eval.wer_impact                                    # acurácia por faixa de WER
```

Não existe áudio real gravado para as frases sintéticas do corpus (só texto), então os
objetivos 2 (n-best) e 3 (impacto do WER) do TCC1 são cumpridos com um pipeline
simulado: sintetiza-se áudio de uma amostra do conjunto de teste via TTS, transcreve-se
com `faster-whisper` e roda-se os 3 classificadores já treinados sobre 4 condições —
texto original (upper bound), transcrição 1-best, e concatenações n-best (N=3, N=5).

| Módulo | Função |
|---|---|
| `sampling.py` | Amostra estratificada por `label + source` do `test.jsonl` |
| `synth.py` | TTS via `edge-tts`, com cache em disco (`outputs/asr_eval/audio/`) |
| `transcribe_nbest.py` | N-best **real** via beam search do `ctranslate2` (não uma aproximação por temperatura) |
| `wer.py` | WER via `jiwer`, texto normalizado dos dois lados |
| `classify.py` | Roda os 3 `ModelSession` sobre cada condição, com o separador correto do tokenizador de cada modelo na concatenação n-best |
| `engine.py` | Orquestra tudo e grava `outputs/asr_eval/{records.jsonl,metrics.json,summary.md}` |

**`transcribe_nbest.py` depende de internals do `faster-whisper`** (`WhisperModel.encode`,
`WhisperModel.get_prompt`, `Tokenizer`, `get_suppressed_tokens`, `pad_or_trim`), não da
API pública `WhisperModel.transcribe()` — que só devolve a melhor hipótese por segmento.
O `ctranslate2.models.Whisper.generate()` interno aceita `num_hypotheses`, confirmado por
`help(ctranslate2.models.Whisper.generate)`. Validado manualmente: para
`"Fecha essa aba, por favor!"`, N=3 devolveu três variações plausíveis de pontuação/
conjugação do beam search real. Se uma versão futura do `faster-whisper` mudar essa API
interna, o fallback documentado é aproximar n-best por amostragem de temperatura
(`best_of` na API pública).

## Validação externa (`src/external_eval/`)

```bash
uv run python -m src.external_eval.run_external_eval                              # os 3 corpora, 1000 exemplos cada
uv run python -m src.external_eval.run_external_eval --corpora coraa --sample-size 100
```

Testa os 3 classificadores contra fala real (não sintética) de três corpora públicos no
Hugging Face Hub — CORAA, NURC-SP, TAGARELA — lendo só a transcrição textual (nenhum
áudio é decodificado; ver docstring de `fetch.py` para o schema de cada corpus e por que
TAGARELA é lido via `pyarrow` direto em vez de `datasets`, evitando a dependência de
decodificação de áudio `torchcodec`). Como nenhum dos três tem anotação DD/NDD, o rótulo
NDD é assumido por suposição de domínio e a métrica é a taxa de falso positivo.

**Resultado**: 20% a 35% de falso positivo em fala real, concentrado em fragmentos
curtos de conversa — evidência de que o classificador usa brevidade da frase como atalho
para DD, um padrão válido no corpus sintético (DD é sempre comando curto) mas que não se
sustenta em fala real (fragmentos curtos e incompletos são comuns e não indicam intenção
de comando). Ver Seção 5.8 do README raiz para a análise completa.

## Convenções do código

- Sem loggers externos (wandb/tensorboard/mlflow) — tudo grava em disco via
  `reporting.py`, sem depender de rede ou conta.
- Sem arquivo de configuração externo (YAML/JSON) para hiperparâmetros de treino — tudo
  é `dataclass` Python em `config.py`, sobrescrevível via CLI. O único YAML do projeto é
  a taxonomia de geração (`src/dataset/prompts/context_schema.yaml`).
- `DD` é sempre a classe positiva (`LABEL2ID = {"NDD": 0, "DD": 1}`) — não inverter; isso
  troca o sentido de precision/recall/EER em todos os relatórios.
- Sem suíte de testes automatizada (`tests/`) — as flags `--allow-cpu`,
  `--max-train-samples`, `--max-eval-samples`, `--max-steps` dos scripts de treino
  servem como smoke test manual.
