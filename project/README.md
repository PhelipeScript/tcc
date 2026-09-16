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
