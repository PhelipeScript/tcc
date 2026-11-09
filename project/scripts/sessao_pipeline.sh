#!/usr/bin/env bash
# Sessão do sistema completo em tempo real: um conjunto de áudio e os 5 modelos em sequência, cada um no seu
# processo (a semente do Whisper vale por processo), seguida da comparação (`ddsd eval comparar`).
#
# Uso: scripts/sessao_pipeline.sh {gravacoes|tts|publico} [modelo ...]
#   padrão: bertimbau baseline_logreg baseline_svm debertinha albertina
#
# - Recusa começar com código não commitado (o log registra o commit) ou na bateria (PERMITIR_BATERIA=1 libera).
# - Um log por modelo em data/logs/pipeline_<conjunto>_<modelo>.log, com energia e código no início e no fim.
# - O `caffeinate -i` impede o Mac de dormir durante as execuções e a comparação.
set -euo pipefail
cd "$(dirname "$0")/.."

usage="uso: scripts/sessao_pipeline.sh gravacoes|tts|publico [modelo ...]"
if [ $# -lt 1 ]; then
  echo "$usage" >&2
  exit 2
fi
dataset="$1"
shift
case "$dataset" in
  gravacoes|tts|publico) ;;
  *) echo "conjunto desconhecido: ${dataset} (${usage})" >&2; exit 2 ;;
esac
if [ $# -gt 0 ]; then
  models=("$@")
else
  models=(bertimbau baseline_logreg baseline_svm debertinha albertina)
fi

if [ -n "$(git status --porcelain -- ddsd config scripts pyproject.toml uv.lock)" ]; then
  echo "há alterações não commitadas no código; commite antes de rodar" >&2
  exit 1
fi
if pmset -g batt | grep -q "Battery Power" && [ "${PERMITIR_BATERIA:-0}" != "1" ]; then
  echo "o Mac está na bateria: ligue na tomada (ou rode com PERMITIR_BATERIA=1)" >&2
  exit 1
fi
if pgrep -qf "LM Studio"; then
  echo "aviso: o LM Studio está aberto e disputa memória e GPU com o Whisper" >&2
fi

energy() { pmset -g batt | tail -n 1 | sed 's/^[[:space:]]*//'; }

for m in "${models[@]}"; do
  code=$(git rev-parse --short HEAD)  # a cada modelo: é o código que o processo vai carregar
  log="data/logs/pipeline_${dataset}_${m}.log"
  echo "$(date '+%H:%M') ${dataset}/${m}: rodando (log em ${log})"
  {
    echo "== início $(date '+%F %T') | $(energy) | código ${code}"
    status=0
    caffeinate -i .venv/bin/ddsd eval pipeline "$dataset" --classificador "$m" || status=$?
    echo "== fim $(date '+%F %T') | $(energy) | saída ${status}"
  } > "$log" 2>&1
  warnings=$(grep -c ' AVI ' "$log" || true)
  errors=$(grep -c ' ERR ' "$log" || true)
  echo "$(date '+%H:%M') ${dataset}/${m}: fim | avisos ${warnings} | erros ${errors}"
  if ! grep -q 'saída 0$' "$log"; then
    echo "a execução de ${m} falhou: veja ${log}" >&2
    exit 1
  fi
done

log="data/logs/comparar_${dataset}.log"
status=0
caffeinate -i .venv/bin/ddsd eval comparar "$dataset" > "$log" 2>&1 || status=$?
grep -E 'mesma transcrição|paridade|comparação em|ERR' "$log" || true
echo "$(date '+%H:%M') ${dataset}: comparação terminada (saída ${status})"
exit "$status"
