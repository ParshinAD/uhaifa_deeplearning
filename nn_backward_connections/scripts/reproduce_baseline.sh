#!/usr/bin/env bash
# Reproduce the Rocket baseline end-to-end:
#   1. ensure pytest      2. run the oracle parity + known-answer gate
#   3. run baseline Rocket on BOTH datasets x 3 seeds (exact scorer)
#   4. aggregate results into experiments/log.md (mean +/- std)
#
# Usage:
#   scripts/reproduce_baseline.sh            # full run (connectome 20k epochs)
#   scripts/reproduce_baseline.sh --quick    # smoke run (30s time limit per run)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

PY="${MFAS_PYTHON:-/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python}"
if [ ! -x "$PY" ]; then PY="python3"; fi

SEEDS=(42 123 999)
QUICK_ARGS=()
if [ "${1:-}" = "--quick" ]; then
  echo ">> QUICK mode: 30s time limit per run"
  QUICK_ARGS=(--time-limit 30)
fi

echo ">> Using interpreter: $PY"
echo ">> Step 1: ensure pytest"
"$PY" -c "import pytest" 2>/dev/null || "$PY" -m pip install -q "pytest>=7,<9"

echo ">> Step 2: oracle gate (parity + known-answer)"
"$PY" -m pytest tests/test_metrics.py -v

echo ">> Step 3: baseline Rocket on both datasets x ${#SEEDS[@]} seeds"
for DS in connectome mouse; do
  for S in "${SEEDS[@]}"; do
    echo "   -- dataset=$DS seed=$S"
    "$PY" -m eval.harness --algo baseline_rocket --dataset "$DS" --seed "$S" \
      --out results/ --config configs/baseline_rocket.yaml ${QUICK_ARGS[@]+"${QUICK_ARGS[@]}"}
  done
done

echo ">> Step 4: aggregate -> experiments/log.md"
"$PY" -m eval.aggregate --glob "results/*.json" --out experiments/log.md

echo ">> DONE. See experiments/log.md and results/*.json"
