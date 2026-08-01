# DeepSeek-V4-Flash-0731 on 2x Intel Arc Pro B70 (llama.cpp SYCL)

Serve **DeepSeek-V4-Flash-0731** (284B params / 13B active MoE, `deepseek4` arch,
Unsloth `UD-Q8_K_XL` GGUF, 162 GB) on **2x Intel Arc Pro B70 Pro GPUs (64 GB VRAM
total) + 128 GB system RAM** — fully resident in VRAM + RAM, with **zero SSD
streaming during inference**.

Measured result: **6.6 tok/s generation, ~12x faster** than the naive
large-context setup (0.56 tok/s), with the entire model off of disk after warmup.

```
+-----------------------+      +-----------------------+
| Arc Pro B70 (GPU0)    |      | Arc Pro B70 (GPU1)    |
| 30.1 / 31.9 GiB used  |      | 30.8 / 31.9 GiB used  |
+----------+------------+      +-----------+-----------+
           | SYCL (oneAPI 2026.0, Level Zero) |
           +----------------+----------------+
                            |
              +-------------v-------------+
              | System RAM: 116 GiB RSS   |
              | (mmap page cache, swap    |
              |  permanently disabled)    |
              +---------------------------+
```

## Quick start (pull, build, use)

```bash
# 1. Pull — the submodule brings the exact llama.cpp source + its MIT license
git clone --recurse-submodules \
  https://github.com/chmihimabb-max/llama.cpp-2x-Arc-B70-Pro-SYCL-DeepSeekV4-Flash-731-Custom
cd llama.cpp-2x-Arc-B70-Pro-SYCL-DeepSeekV4-Flash-731-Custom

# 2. Get the model weights (162 GB, not distributed by this repo)
hf download unsloth/DeepSeek-V4-Flash-0731-GGUF --include "UD-Q8_K_XL/*"

# 3. Build (needs Intel oneAPI 2026.0 installed)
scripts/build.sh

# 4. Tune the system (swap off, readahead 0 — see below)
sudo scripts/system-tuning.sh

# 5. Serve
scripts/serve.sh

# 6. Benchmark
python3 scripts/bench.py
```

## Why this repo exists

The model (162 GB) *almost* fits in 64 GB VRAM + 128 GB RAM. The default instinct
— crank context to 512K — backfires: KV cache + compute buffers crowd weights off
the GPUs, leaving ~131 GiB on the CPU side, which is **more than RAM**. The kernel
then evicts and re-faults model pages from SSD on every token, collapsing
throughput to ~0.5 tok/s and thrashing the SSD.

The fix is arithmetic, not exotic: shrink context to 256K so the GPUs can hold
their full share, disable swap so the kernel can never page the model out, and
kill block-level readahead which amplifies scattered MoE expert faults ~3x.

## Hardware / software stack

| Component | Value |
|---|---|
| CPU | Intel Core i7-12700KF (8P+4E / 20 threads) |
| Motherboard | ASUS PRIME Z690-P — BIOS x8/x8 bifurcation enabled for future expansion; current wiring: GPU0 on CPU Gen4 x16, GPU1 on chipset Gen4 x4 (full topology in [docs/environment.md](docs/environment.md)) |
| GPUs | 2x Intel Arc Pro B70 (32 GiB VRAM each, `xe` kernel driver) |
| RAM | 128 GB (4x 32 GB G.Skill DDR5, 123.3 GiB usable) |
| Disk | Kingston 2 TB NVMe SSD (model on ext4) |
| OS | Ubuntu 26.04 LTS, kernel 7.0.0-28, oneAPI 2026.0, Level Zero 26.22 |
| llama.cpp | vendored as git submodule in `vendor/llama.cpp`, pinned @ `876a43211` — **unmodified upstream**. `deepseek4` support landed in PR [#24162](https://github.com/ggml-org/llama.cpp/pull/24162); no fork needed |
| Model | [`unsloth/DeepSeek-V4-Flash-0731-GGUF`](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF) `UD-Q8_K_XL` (5 shards, 162 GB) |

## Build

`scripts/build.sh` reproduces the exact verified build — its flags match the
as-built `CMakeCache.txt` on the benchmark machine verbatim. Full toolchain
versions and the verified configure record:
[docs/environment.md](docs/environment.md).

Three DeepSeek-V4-specific fixes beyond a standard SYCL build:

1. `-DGGML_SCHED_MAX_SPLIT_INPUTS=256` — the compressed-attention graph exceeds
   the default 30-input split limit (`GGML_ASSERT` at load without it).
2. `-O0` — icpx 2026.0 segfaults compiling flash-attention template instantiations
   at any higher opt level. DSv4 uses FlashMLA, so the slower FA kernels cost
   nothing at runtime.
3. `-liomp5 -lsvml -lirng -limf -lintlc` + `MKL_DIR` — Intel-compiler objects need
   Intel OpenMP/math runtimes that GNU ld can't resolve on its own.

`LD_LIBRARY_PATH` must include **both** `/opt/intel/oneapi/compiler/2026.0/lib`
(libsvml, libdnnl, libiomp5) and `/opt/intel/oneapi/2026.0/lib` (libumf — without
it the Level Zero adapter vanishes and SYCL sees zero platforms).

## Serve

`scripts/serve.sh`:

```bash
llama-server \
  -m .../DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf \
  --jinja \
  -c 262144 \
  --device SYCL0,SYCL1 \
  --fit on \
  -fa off \
  --port 58190 \
  --host 0.0.0.0 \
  --temp 1.0
```

Notes:

- **`-c 262144` is the residency ceiling on 128 GB RAM.** 512K works but
  SSD-streams; 1M OOMs (37 GiB compute buffer > 32 GiB card). DSv4 also misbehaves
  below 256K, so 262144 is the sweet spot, not just a fallback.
- `--fit on` balances both cards (~30.7/31.9 GiB each). At 512K it gives up and
  leaves GPU1 nearly empty — check with a Level Zero sysman tool, because on the
  `xe` driver `intel_gpu_top`, sysfs `mem_info`, and `/proc/*/fdinfo` are all
  blind to VRAM. (`zemem` source included in `scripts/zemem.c`.)
- `-fa off` — flash attention is unsupported for `deepseek4` on SYCL; `-fa on`
  silently reassigns layers to CPU.
- No `-ctk/-ctv` — MLA compresses KV natively; quantization is unsupported and
  unnecessary.
- **Do not use `--no-mmap`** — it forces 162 GB into 128 GB RAM and swaps.

## System tuning (the important part)

`scripts/system-tuning.sh`:

1. **Disable swap permanently.** With swap on, the kernel paged 15 GiB out under
   pressure. Comment the swap entries in `/etc/fstab`, `sudo swapoff -a`,
   `sudo systemctl daemon-reload`. (Files can then be deleted to reclaim disk.)
2. **Zero block readahead on the model disk.**
   `echo 0 | sudo tee /sys/block/nvme1n1/queue/read_ahead_kb`
   Default 128 KB readahead amplifies scattered mmap expert faults ~3x
   (6.6 -> 2.2 GiB SSD per 400 tokens measured). Not persistent — use a udev
   rule if you want it permanent.

## Benchmark (measured 2026-07-31)

Method: `scripts/bench.py` — 400-token generations via `/v1/chat/completions`,
SSD reads measured from `/proc/<pid>/io read_bytes` deltas around each run.

| Config | Gen tok/s | GPU0/GPU1 used | RSS | SSD reads per 400 tok |
|---|---|---|---|---|
| 512K, swap on, readahead 128 | **0.56** | 18.8 / 1.1 GiB | 117 GiB (overcommitted) | continuous streaming |
| 256K, swap off, readahead 128, run 1 | 4.06 | 30.1 / 30.8 GiB | 92 GiB | 47.1 GiB (cold experts) |
| 256K, run 2 (new prompt) | 5.76 | " | 116 GiB | 11.5 GiB |
| 256K, run 3 (repeat prompt) | 6.03 | " | 116 GiB | 6.6 GiB |
| 256K, run 4 (readahead=0) | **6.58** | " | 116 GiB | **2.2 GiB** |

- SSD reads decay as cold MoE experts fault into page cache; after warmup the
  residual is ~2 GiB/400 tok (~37 MB/s) and shrinking — the model effectively
  runs out of VRAM + RAM.
- Warm prompt processing: ~11.7 tok/s.
- **~12x generation speedup** from the 512K baseline.

Reproduce:

```bash
python3 scripts/bench.py --url http://127.0.0.1:58190 --runs 4 --max-tokens 400
```

## Verification cheatsheet

```bash
# true per-card VRAM on the xe driver (Level Zero sysman)
./zemem
# server resident set + disk reads
ps -o rss= -p $(pgrep -x llama-server)
grep read_bytes /proc/$(pgrep -x llama-server)/io
# memory headroom
free -h
```

## Gotchas

- A broken conda shell hook can corrupt the environment for oneAPI binaries.
  Launch with `env -i PATH="$PATH" HOME="$HOME" bash --norc --noprofile -c '...'`
  (the serve script does this).
- `sycl-ls` showing "No platforms found" = missing `libumf.so.1` in
  `LD_LIBRARY_PATH` (add `/opt/intel/oneapi/2026.0/lib`).
- 1M context fails with `failed to allocate SYCL0 buffer of size 37059346688` —
  the compute buffer, not the KV cache, is the bottleneck (`--no-kv-offload`
  doesn't help).
- RAM math for other budgets: model 162 GB; need VRAM_offload + RAM >= ~169 GB.
  With 192 GB RAM, 512K becomes fully SSD-free too.

## Licensing

- This repo's scripts and docs: **MIT** — see [LICENSE](LICENSE).
- `vendor/llama.cpp`: git submodule referencing
  [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), **MIT**,
  unmodified; its license text ships inside the submodule checkout. The MIT
  license permits this redistribution/reference as long as the notice is
  retained — see [NOTICE.md](NOTICE.md).
- Model weights are **not** redistributed here; download from Hugging Face and
  use under the model's own license.
