#!/usr/bin/env bash
# Serve DeepSeek-V4-Flash-0731 UD-Q8_K_XL with DSpark speculative decoding.
# Draft model runs out of SYSTEM MEMORY (-ngld 0), target stays on 2x Arc B70.
# 2x Intel Arc Pro B70 (64 GiB VRAM) + 128 GiB RAM. Swap must be off.
#
# Requires a llama.cpp build with DeepSeekV4 MTP + DSpark (#25784, >= 2026-08-06)
# Prereqs: scripts/build.sh on updated vendor, or LLAMA_SERVER pointing at a
#          newer build (e.g. ~/llama.cpp/build/bin/llama-server).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_SERVER="${LLAMA_SERVER:-/home/mike/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-/run/media/mike/WDC 1TB Volume/models/models--unsloth--DeepSeek-V4-Flash-0731-GGUF/snapshots/ce2d0d94e35d96e9412afee41b9b32b21529bf1b/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf}"
DRAFT="${DRAFT:-/run/media/mike/WDC 1TB Volume/models/dspark-DeepSeek-V4-Flash-0731-Q8_0/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf}"
PORT="${PORT:-58190}"
CTX="${CTX:-262144}"   # 256K = residency ceiling on 128GB RAM, same as serve.sh

# env -i + bash --norc: a broken conda shell hook on this box corrupts the
# environment for oneAPI binaries. Keep the launch environment clean.
exec env -i PATH="$PATH" HOME="$HOME" \
  LD_LIBRARY_PATH="/opt/intel/oneapi/compiler/2026.0/lib:/opt/intel/oneapi/2026.0/lib" \
  bash --norc --noprofile -c \
  "'$LLAMA_SERVER' \
    -m '$MODEL' \
    -md '$DRAFT' \
    --spec-type draft-dspark \
    --spec-draft-n-max 3 \
    -ngld 0 \
    --jinja \
    -c $CTX \
    --device SYCL0,SYCL1 \
    --fit on \
    -fa off \
    --port $PORT \
    --host 0.0.0.0 \
    --temp 1.0"
