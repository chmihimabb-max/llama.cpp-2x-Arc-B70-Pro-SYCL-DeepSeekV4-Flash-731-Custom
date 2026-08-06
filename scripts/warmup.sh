#!/usr/bin/env bash
# Warm the page cache with the model (+ optional DSpark draft) so mmap'd pages
# fault from RAM instead of the SSD. Run AFTER the server has started (the
# server's mmap benefits immediately — page cache is global). First requests
# on a cold cache are 3-10x slower than steady state (bench.py runs 1-2 vs
# runs 3-4: 4.7 -> 6.9 tok/s, SSD 52 -> 7 GiB/400 tok, measured 2026-08-06).
#
# NOTE: the model (162 GB) exceeds RAM (123 GB), so the warm oversubscribes the
# cache; LRU keeps ~110 GB. Steady-state decode still rotates ~7-9 GiB/400 tok
# from disk on this box. The warm's job is making the *common* experts resident
# so decode runs at GPU speed instead of SSD-fault speed.
#
# Usage: scripts/warmup.sh
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/run/media/mike/WDC 1TB Volume/models/models--unsloth--DeepSeek-V4-Flash-0731-GGUF/snapshots/ce2d0d94e35d96e9412afee41b9b32b21529bf1b/UD-Q8_K_XL}"
DRAFT="${DRAFT:-/run/media/mike/WDC 1TB Volume/models/dspark-DeepSeek-V4-Flash-0731-Q8_0/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf}"

echo "== warming model shards ($(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1) =="
cat "$MODEL_DIR"/*.gguf > /dev/null

if [ -f "$DRAFT" ]; then
  echo "== warming dspark draft ($(du -sh "$DRAFT" | cut -f1) =="
  cat "$DRAFT" > /dev/null
else
  echo "(no draft file at $DRAFT — skipping)"
fi

echo "WARMED"
