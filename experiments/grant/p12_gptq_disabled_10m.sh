#!/usr/bin/env bash
# Track P — GPTQ Hessian path skipped; still int6 serialize via mixed_quantize. Compare to p01/p13.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_track_p_env.sh"
cd "$REPO_ROOT"

export DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
export RUN_ID="${RUN_ID:-p12_gptq_disabled_10m}"
export SEED="${SEED:-1337}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export VOCAB_SIZE="${VOCAB_SIZE:-4096}"
export GPTQ_ENABLED=0

torchrun --standalone --nproc_per_node=1 "$TRAIN_PY"

echo "Log: $REPO_ROOT/logs/${RUN_ID}.txt"
