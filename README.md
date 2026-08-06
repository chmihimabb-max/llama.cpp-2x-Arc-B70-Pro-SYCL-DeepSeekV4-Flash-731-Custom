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

# 4. Tune the system (swap off, readahead 2 MiB — see below)
sudo scripts/system-tuning.sh

# 5. Serve
scripts/serve.sh

# 5b. Recommended: run as a systemd service with memory.min pinning instead —
#     survives reboots, warms the cache inside the same cgroup, and the kernel
#     guarantees the model's ~110 GiB CPU-side pages are never evicted
sudo cp systemd/llama-server.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now llama-server

# 6. Warm the page cache (first requests are 3-10x slower on a cold cache)
scripts/warmup.sh   # run after serve starts; page cache is global
   # (unnecessary with 5b — the service's ExecStartPost does it in-cgroup)

# 7. Benchmark
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
| OS | Ubuntu 26.04 LTS, kernel 7.0.0-28, `xe` driver + GuC 70.58.0 / HuC 8.2.10 firmware, Level Zero 26.22, oneAPI 2026.0 |
| llama.cpp | vendored as git submodule in `vendor/llama.cpp`, pinned @ `6a32c29a7` — **unmodified upstream**. `deepseek4` support landed in PR [#24162](https://github.com/ggml-org/llama.cpp/pull/24162); DeepSeekV4 MTP + DSpark in [#25784](https://github.com/ggml-org/llama.cpp/pull/25784); no fork needed |
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
- `-fa off` — flash attention is unsupported for `deepseek4` on SYCL. On the
  current build `-fa on` **segfaults in `libintlc.so.5` on the first request**
  (measured 2026-08-06); on the older build it silently reassigned layers to
  CPU. Keep it off.
- No `-ctk/-ctv` — MLA compresses KV natively; quantization is unsupported and
  unnecessary.
- **Do not use `--no-mmap`** — the SYCL backend stages all tensors in one
  contiguous host buffer before offloading (`failed to allocate SYCL_Host buffer
  of size 141383696384`), so it needs ~135 GB RAM on top of a 128 GB machine.
  Tested 2026-08-01; clean exit, no GPU hang. mmap is mandatory.

## System tuning (the important part)

`scripts/system-tuning.sh`:

1. **Disable swap permanently.** With swap on, the kernel paged 15 GiB out under
   pressure. Comment the swap entries in `/etc/fstab`, `sudo swapoff -a`,
   `sudo systemctl daemon-reload`. (Files can then be deleted to reclaim disk.)
2. **Set block readahead on the model disk to 2 MiB.**
   `sudo blockdev --setra 4096 /dev/nvme0n1` (sysfs: `read_ahead_kb` = 2048).
   Do NOT zero it: with readahead=0 every mmap page fault is a lone 4 KiB read,
   so the initial load runs latency-bound at ~240 MiB/s while the drive does
   1.75 GiB/s O_DIRECT (measured). The old "0" recipe only mattered in the 512K
   SSD-streaming regime (128 KB readahead amplified scattered expert faults ~3x,
   6.6 -> 2.2 GiB SSD per 400 tokens); at 256K residency the only re-reads are
   the ~6 GiB/run page-cache rotation, and readahead makes those ~4-7 s of
   stalls instead of ~25 s. Persisted by a udev rule
   (`99-llm-readahead.rules`, created by `system-tuning.sh`).

## Pinning the model in RAM (memory.min — the clean way)

The model's CPU-side portion (~100 GiB of 162 GB) lives in the OS page cache,
which is reclaimable by design — any memory pressure (another model, a big
download, the DSpark draft) evicts model pages and decode fault-trips to the
SSD. mlock is not an option: llama.cpp's `--load-mode mlock` locks the entire
162 GB mapping, which exceeds the 123 GB RAM (documented OOM). `--no-mmap`
fails on SYCL (134 GB host staging buffer).

The clean mechanism is cgroup v2 `memory.min` (systemd `MemoryMin=`): the
kernel will not reclaim pages charged to the unit's cgroup while usage is at or
below the floor. `systemd/llama-server.service` sets `MemoryMin=110G` — the
model's observed CPU-side working set (peak RSS 108.5 GiB measured 2026-08-06;
162 GB total minus ~61 GB VRAM) — leaving ~13 GiB for the rest of the system
(today: ~6-10 GiB in use, safe for typical use; a second big model alongside
will NOT fit without lowering the floor). The guarantee is hard: under pressure
the kernel OOM-kills unprotected processes rather than violate it, so don't
raise the floor without headroom.

The floor is live-writable without a restart:
`sudo bash -c 'echo 118111600640 > /sys/fs/cgroup/system.slice/llama-server.service/memory.min'`
(118111600640 = 110 GiB; do this AND update the unit file so a later restart
keeps it).

**Critical detail — page-charge ownership:** page cache is charged to the
cgroup that faults it in. The warmup must run INSIDE the server's cgroup,
which is why the unit's `ExecStartPost` runs `scripts/warmup.sh` — a manual
`cat` from another terminal charges those pages to that terminal's cgroup and
they are NOT protected.

Verify live:

```bash
systemctl cat llama-server | grep MemoryMin
cat /sys/fs/cgroup/system.slice/llama-server.service/memory.min  # 107374182400
```

## DSpark speculative decoding (tested 2026-08-06)

DeepSeek's DSpark drafter (block-parallel, 5-token blocks, shares the target's
embeddings/lm_head) is supported by llama.cpp as of #25784 and ships as a
ready-made GGUF from Unsloth (`dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf`,
10.9 GB; BF16 variant under `dspark/`). `scripts/serve-dspark.sh` runs it
**from system RAM** (`-ngld 0`) alongside the plain serve.sh config.

**Result: net loss on this machine — do not enable.** Measured with
`scripts/bench.py`, warm cache:

| Config | Gen tok/s | SSD reads per 400 tok |
|---|---|---|
| Plain, new build | **6.76-6.94** | 7-9 GiB |
| + DSpark draft on CPU (`-ngld 0`) | **0.24** | 42 GiB |

Two independent reasons:

1. The CPU-resident drafter serializes every decode step — the GPU target idles
   while each 3-token draft block is computed on CPU.
2. The draft's 10.9 GB breaks the 128 GB RAM residency ceiling: RSS drops
   99 -> 64 GiB and the model thrashes the SSD (42 GiB per 400 tokens).

GPU-side drafting (`-ngld 99`) is impossible here — both cards are ~full at
256K ctx (0.5-2 GiB free). It would need freed VRAM or more RAM. Note the docs
say spec paths want `-fa on`, but that segfaults on DSV4/SYCL; the current
build activates `draft-dspark` fine with `-fa off` (block_size=5, noise token
128799) — the blocker is purely the drafter economics.

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

### Update 2026-08-06 — build b10297 (`6a32c29a7`)

Rebuilt with the same flags. Steady state matches the old build (runs 3-4 are
warm — run 1 is the "first after launch" case):

| run | Gen tok/s | prompt t/s | SSD read GiB |
|---|---|---|---|
| 1 (post-launch) | 4.73 | 2.87 | 51.9 |
| 2 | 5.68 | 3.81 | 27.4 |
| 3 (repeat prompt) | 6.94 | 11.12 | 7.1 |
| 4 | 6.76 | 9.61 | 8.6 |

Warmup procedure: `scripts/warmup.sh` — `cat` all model shards into the page
cache after the server starts (page cache is global, the server's mmap benefits
immediately). Without it the first requests run at SSD-fault speed; steady
state is 6.8-6.9 tok/s with ~7-9 GiB/400 tok residual rotation.

### Update 2026-08-06 — memory.min pinning proven under pressure

After moving the server to the systemd unit (`systemd/llama-server.service`,
`MemoryMin=100G`), the first bench run reads 111.8 GiB from SSD — the migration
cost: model pages previously charged to the old terminal scope were evicted as
the service's protected cgroup grew, and the server re-faulted them into its
own (now-protected) cgroup. Steady state returns by run 2-3.

Then, with a 12 GiB anonymous allocation held in an unprotected cgroup (swap
off — anon is unreclaimable, so the kernel MUST evict file pages):

| run (under 12 GiB pressure) | Gen tok/s | prompt t/s | SSD read GiB |
|---|---|---|---|
| 1 | 6.71 | 11.58 | 9.2 |
| 2 | 6.58 | 10.59 | 10.0 |
| 3 | 6.92 | 13.17 | 6.5 |

Identical to the no-pressure baseline (6.80 / ~8 GiB). Cgroup accounting
confirms it: server cgroup stayed at 103.9 GiB (floor was 100G at the time,
raised to 110G later the same day) while the unprotected old terminal scope
dropped 115.2 -> 13.5 GiB — the kernel satisfied the pressure from unprotected
memory. The model cannot be evicted anymore.

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
- `-fa on` on DeepSeek-V4/SYCL segfaults in `libintlc.so.5` on the first
  request (builds >= b10297). Keep `-fa off` — see Serve notes.

## Licensing

- This repo's scripts and docs: **MIT** — see [LICENSE](LICENSE).
- `vendor/llama.cpp`: git submodule referencing
  [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), **MIT**,
  unmodified; its license text ships inside the submodule checkout. The MIT
  license permits this redistribution/reference as long as the notice is
  retained — see [NOTICE.md](NOTICE.md).
- Model weights are **not** redistributed here; download from Hugging Face and
  use under the model's own license.
