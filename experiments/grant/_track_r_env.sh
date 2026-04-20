# Shared paths for Track R (repo-root train_gpt.py — no FlashAttention 3).
# shellcheck shell=bash
#
# Data:
#   python3 data/cached_challenge_fineweb.py --variant sp1024
# Byte-level variant (optional):
#   python3 data/cached_challenge_fineweb.py --variant byte260

GRANT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$GRANT_DIR/../.." && pwd)"
export TRAIN_R_PY="$REPO_ROOT/train_gpt.py"
