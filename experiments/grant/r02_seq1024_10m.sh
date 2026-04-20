#!/usr/bin/env bash
# Track R — default context 1024 (baseline shape). Data: cached_challenge_fineweb.py --variant sp1024
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_track_r_env.sh"
cd "$REPO_ROOT"

export DATA_PATH="${DATA_PATH:-$REPO_ROOT/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$REPO_ROOT/data/tokenizers/fineweb_1024_bpe.model}"
export RUN_ID="${RUN_ID:-r02_seq1024_10m}"
export SEED="${SEED:-1337}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export VOCAB_SIZE=1024
export TRAIN_SEQ_LEN=1024
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-2000}"

torchrun --standalone --nproc_per_node=1 "$TRAIN_R_PY"

echo "Log: $REPO_ROOT/logs/${RUN_ID}.txt"
