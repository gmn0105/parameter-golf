#!/usr/bin/env bash
# Track R — byte-level tokenizer (260). Download: python3 data/cached_challenge_fineweb.py --variant byte260
# If vocab/tokenizer path errors, set TOKENIZER_PATH to the .model file under data/tokenizers/ from your manifest.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_track_r_env.sh"
cd "$REPO_ROOT"

export DATA_PATH="${DATA_PATH:-$REPO_ROOT/data/datasets/fineweb10B_byte260}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$REPO_ROOT/data/tokenizers/fineweb_260_bpe.model}"
export RUN_ID="${RUN_ID:-r01_byte260_10m}"
export SEED="${SEED:-1337}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export VOCAB_SIZE="${VOCAB_SIZE:-260}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-2000}"

torchrun --standalone --nproc_per_node=1 "$TRAIN_R_PY"

echo "Log: $REPO_ROOT/logs/${RUN_ID}.txt"
