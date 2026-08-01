#!/usr/bin/env bash
# Serve DeepSeek-V4-Flash-0731 UD-Q8_K_XL fully resident in VRAM + RAM.
# 2x Intel Arc Pro B70 (64 GiB VRAM) + 128 GiB RAM. Swap must be off.
set -euo pipefail

MODEL="${MODEL:-$HOME/.cache/huggingface/hub/models--unsloth--DeepSeek-V4-Flash-0731-GGUF/snapshots/ce2d0d94e35d96e9412afee41b9b32b21529bf1b/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf}"
PORT="${PORT:-58190}"
CTX="${CTX:-262144}"   # 256K = residency ceiling on 128GB RAM. 512K SSD-streams; 1M OOMs.

# env -i + bash --norc: a broken conda shell hook on this box corrupts the
# environment for oneAPI binaries. Keep the launch environment clean.
exec env -i PATH="$PATH" HOME="$HOME" \
  LD_LIBRARY_PATH="/opt/intel/oneapi/compiler/2026.0/lib:/opt/intel/oneapi/2026.0/lib" \
  bash --norc --noprofile -c \
  "'$HOME/llama.cpp/build/bin/llama-server' \
    -m '$MODEL' \
    --jinja \
    -c $CTX \
    --device SYCL0,SYCL1 \
    --fit on \
    -fa off \
    --port $PORT \
    --host 0.0.0.0 \
    --temp 1.0"
