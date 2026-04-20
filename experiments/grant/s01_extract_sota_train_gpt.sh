#!/usr/bin/env bash
# Track S — extract inner source from LZMA-packed SOTA train_gpt.py (research only; not a submission artifact).
set -euo pipefail
GRANT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$GRANT_DIR/../.." && pwd)"
DEFAULT_SRC="$REPO_ROOT/records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/train_gpt.py"
SRC="${1:-$DEFAULT_SRC}"
OUT="${2:-$REPO_ROOT/experiments/sota_decompressed/train_gpt_extracted.py}"

python3 "$REPO_ROOT/scripts/decompress_record_train_gpt.py" "$SRC" -o "$OUT"
echo "Extracted: $OUT"
echo "Run with torchrun only after inspecting the extracted script (deps, paths, WORLD_SIZE rules)."
