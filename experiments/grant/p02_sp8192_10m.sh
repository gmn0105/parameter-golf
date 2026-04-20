#!/usr/bin/env bash
# Track P — SP8192 vocab/data. If OOM, rerun with: TRAIN_BATCH_TOKENS=393216 ./p02_sp8192_10m.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_track_p_env.sh"
cd "$REPO_ROOT"

export DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
export RUN_ID="${RUN_ID:-p02_sp8192_10m}"
export SEED="${SEED:-1337}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export VOCAB_SIZE=8192
export GPTQ_ENABLED=1
# Default record batch is 2048*48*8; reduce if 8192 vocab exhausts H100 memory:
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}"

torchrun --standalone --nproc_per_node=1 "$TRAIN_PY"

echo "Log: $REPO_ROOT/logs/${RUN_ID}.txt"
