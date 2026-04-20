#!/usr/bin/env bash
# Track P — MUON_WD=0.08, EMA_DECAY=0.997 (5 min, sparse val).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_track_p_env.sh"
cd "$REPO_ROOT"

export DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
export RUN_ID="${RUN_ID:-p07_wd0p08_ema997_5m}"
export SEED="${SEED:-1337}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-300}"
export VOCAB_SIZE="${VOCAB_SIZE:-4096}"
export MUON_WD=0.08
export EMA_DECAY=0.997
export VAL_LOSS_EVERY=999999
export GPTQ_ENABLED=1

torchrun --standalone --nproc_per_node=1 "$TRAIN_PY"

echo "Log: $REPO_ROOT/logs/${RUN_ID}.txt"
