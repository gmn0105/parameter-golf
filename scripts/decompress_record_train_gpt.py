#!/usr/bin/env python3
"""Extract inner Python source from a one-line LZMA+b85 packed train_gpt.py (Track S / research only)."""

from __future__ import annotations

import argparse
import base64
import lzma
import sys
from pathlib import Path


def decompress_record_train_gpt(src: Path) -> bytes:
    text = src.read_text(encoding="utf-8")
    marker = 'b85decode("'
    i = text.find(marker)
    if i < 0:
        raise ValueError(f"Could not find {marker!r} in {src}")
    start = i + len(marker)
    end = text.find('"),format', start)
    if end < 0:
        raise ValueError(f"Could not find end marker after b85 payload in {src}")
    raw = base64.b85decode(text[start:end])
    return lzma.decompress(raw, format=lzma.FORMAT_RAW, filters=[{"id": lzma.FILTER_LZMA2}])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "input",
        type=Path,
        help="Packed train_gpt.py (e.g. records/.../train_gpt.py with lzma+b85)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("experiments/sota_decompressed/train_gpt_extracted.py"),
        help="Where to write extracted source",
    )
    args = p.parse_args()
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    data = decompress_record_train_gpt(args.input)
    out.write_bytes(data)
    print(f"Wrote {len(data)} bytes to {out.resolve()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
