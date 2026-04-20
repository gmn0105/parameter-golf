#!/usr/bin/env bash
# Track P — QK-gain sweep (5 min, sparse val). Use VOCAB 4096; override VOCAB_SIZE after p01/p02 if needed.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_track_p_env.sh"
cd "$REPO_ROOT"

export DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
export RUN_ID="${RUN_ID:-p03_qk_gain_4p0_5m}"
export SEED="${SEED:-1337}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-300}"
export VOCAB_SIZE="${VOCAB_SIZE:-4096}"
export QK_GAIN_INIT=4.0
export VAL_LOSS_EVERY=999999
export GPTQ_ENABLED=1

torchrun --standalone --nproc_per_node=1 "$TRAIN_PY"

echo "Log: $REPO_ROOT/logs/${RUN_ID}.txt"
