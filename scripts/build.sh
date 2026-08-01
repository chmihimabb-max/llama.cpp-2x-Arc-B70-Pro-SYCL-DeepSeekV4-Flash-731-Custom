#!/usr/bin/env bash
# Build llama.cpp (upstream master) with SYCL for DeepSeek-V4 on Intel Arc.
# Tested with oneAPI 2026.0, llama.cpp @ 876a43211.
set -euo pipefail

LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"

cd "$LLAMA_DIR"
# Pin to the tested commit (any master with PR #24162 works)
# git checkout 876a43211

mkdir -p build && cd build

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
