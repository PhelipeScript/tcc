#!/usr/bin/env bash
# Treino completo do experimento: condição principal (3 sementes × 3 modelos) + controle (1 semente).
# Idempotente: pula execuções concluídas e retoma as interrompidas. Rode destacado:
#   nohup caffeinate -i scripts/treino_completo.sh > data/logs/treino_completo.log 2>&1 &
# Mantenha o Mac na tomada e com a tampa aberta.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1

test -f data/corpus/frozen/manifest.json || { echo "corpus principal não congelado (ddsd corpus freeze)"; exit 1; }
test -f data/corpus/control/frozen/manifest.json || { echo "corpus controle não congelado"; exit 1; }

echo "== $(date '+%F %T') condição principal"
.venv/bin/ddsd train all --seeds 42 43 44 --skip-done
status_main=$?

echo "== $(date '+%F %T') condição-controle (sem mitigação)"
.venv/bin/ddsd train all --seeds 42 --skip-done \
  --data-dir data/corpus/control/frozen/splits --output-dir models/controle
status_ctrl=$?

echo "== $(date '+%F %T') fim (principal=$status_main, controle=$status_ctrl)"
exit $(( status_main || status_ctrl ))
