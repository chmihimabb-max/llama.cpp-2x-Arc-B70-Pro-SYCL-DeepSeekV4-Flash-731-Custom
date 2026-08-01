#!/usr/bin/env bash
# Build the vendored llama.cpp (pinned @ 876a43211, includes deepseek4 support)
# with SYCL for DeepSeek-V4 on Intel Arc. Tested with oneAPI 2026.0.
#
# Prereq: clone with submodules —
#   git clone --recurse-submodules <this-repo>
# or after the fact:
#   git submodule update --init --recursive
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="${LLAMA_DIR:-$REPO_ROOT/vendor/llama.cpp}"

if [ ! -f "$LLAMA_DIR/CMakeLists.txt" ]; then
  echo "vendor/llama.cpp is empty — run: git submodule update --init --recursive" >&2
  exit 1
fi

cd "$LLAMA_DIR"
mkdir -p build && cd build

# DeepSeek-V4-specific flags (see README for the why):
#   GGML_SCHED_MAX_SPLIT_INPUTS=256 — compressed-attention graph exceeds the
#                                     default 30-input split limit
#   -O0                             — icpx 2026.0 segfaults on FA template
#                                     instantiation otherwise (runtime-neutral:
#                                     DSv4 uses FlashMLA, not these kernels)
#   -liomp5 -lsvml -lirng -limf -lintlc — Intel OpenMP/math runtimes GNU ld
#                                     can't resolve on its own
cmake .. \
  -DGGML_SYCL=ON \
  -DGGML_SYCL_F16=ON \
  -DCMAKE_C_COMPILER=/opt/intel/oneapi/compiler/2026.0/bin/icx \
  -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2026.0/bin/icpx \
  -DCMAKE_CXX_FLAGS="-O0 -DGGML_SCHED_MAX_SPLIT_INPUTS=256" \
  -DCMAKE_BUILD_TYPE=Release \
  -DMKL_DIR=/opt/intel/oneapi/2026.0/lib/cmake/mkl \
  -DCMAKE_EXE_LINKER_FLAGS="-L/opt/intel/oneapi/compiler/2026.0/lib -liomp5 -lsvml -lirng -limf -lintlc" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L/opt/intel/oneapi/compiler/2026.0/lib -liomp5 -lsvml -lirng -limf -lintlc"

# LD_LIBRARY_PATH is needed DURING the build: cmake runs llama-ui-embed as a build step.
export LD_LIBRARY_PATH="/opt/intel/oneapi/compiler/2026.0/lib:/opt/intel/oneapi/2026.0/lib:${LD_LIBRARY_PATH:-}"
cmake --build . --config Release -j"$(nproc)"

echo "Built: $LLAMA_DIR/build/bin/llama-server"
