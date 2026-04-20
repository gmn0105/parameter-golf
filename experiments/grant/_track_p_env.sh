# Shared env for Track P (readable GPTQ+Muon+EMA record).
# shellcheck shell=bash
#
# Data (from repo root):
#   MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
#     python3 data/cached_challenge_fineweb.py --variant sp4096
#   python3 data/cached_challenge_fineweb.py --variant sp8192
#
# FlashAttention 3 — wheel must match your CUDA + PyTorch build. Example (CUDA 12.8, torch 2.9.1):
#   pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
# If import fails, use Track R (root train_gpt.py) or install a matching wheel from the FA3 project.

GRANT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$GRANT_DIR/../.." && pwd)"
export TRAIN_PY="$REPO_ROOT/records/track_10min_16mb/2026-04-01_Vocab4096_MLPMult4_WD085/train_gpt.py"
