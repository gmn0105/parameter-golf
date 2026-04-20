#!/usr/bin/env bash
# Track P — baseline competitive proxy, SP4096, 10 min wall clock, GPTQ on (default).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_track_p_env.sh"
cd "$REPO_ROOT"

export DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
export RUN_ID="${RUN_ID:-p01_baseline_sp4096_10m}"
export SEED="${SEED:-1337}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export VOCAB_SIZE=4096
export GPTQ_ENABLED=1

torchrun --standalone --nproc_per_node=1 "$TRAIN_PY"

echo "Log: $REPO_ROOT/logs/${RUN_ID}.txt"
