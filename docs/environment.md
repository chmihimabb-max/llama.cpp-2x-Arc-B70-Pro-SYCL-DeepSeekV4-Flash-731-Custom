# Build & runtime environment (as measured 2026-07-31 / 2026-08-01)

Everything below was read directly off the machine that produced the benchmark
numbers. The build commands in `scripts/build.sh` are not a reconstruction —
they match the as-built configuration recorded in `build/CMakeCache.txt`
verbatim (excerpt at the bottom).

## Hardware

| Component | Detail |
|---|---|
| CPU | Intel Core i7-12700KF (Alder Lake, 8P+4E cores / 20 threads, up to 5.0 GHz) |
| Motherboard | ASUS PRIME Z690-P (BIOS: AMI 1620, 2022-08-12) |
| RAM | 128 GB — 4x 32 GB G.Skill DDR5 (F5-5600J3636D32G, rated 5600 MT/s, configured 4000 MT/s) |
| GPU 0 | Intel Arc Pro B70 (Battlemage G31), 32 GiB VRAM, PCIe **Gen4 x16** (CPU-attached) |
| GPU 1 | Intel Arc Pro B70 (Battlemage G31), 32 GiB VRAM, PCIe **Gen4 x4** (chipset-attached) |
| Model disk | Kingston SNV2S2000G 2 TB NVMe (root fs + model shards) |
| Extra disks | WD Blue SN570-class 1 TB NVMe, WD 1 TB HDD, Seagate 4 TB USB |

### PCIe topology (lspci -t, as wired during the benchmark)

```
CPU (i7-12700KF)
├─ 00:01.0 root port — 12th Gen Core PCIe x16 Controller [8086:460d]
│    LnkCap Gen5 x16 · LnkSta Gen4 x16
│    └─ 01:00.0 [8086:e2ff] — GPU card 1 onboard PCIe switch (upstream)
│         LnkCap Gen4 x16 · LnkSta Gen4 x16
│         ├─ 02:01.0 → 03:00.0  Arc Pro B70 GPU die
│         └─ 02:02.0 → 04:00.0  HDMI audio
│
└─ PCH (Z690) 00:1b.4 root port #21 [8086:7ac4]
     LnkCap Gen4 x4 · LnkSta Gen4 x4
     └─ 08:00.0 [8086:e2ff] — GPU card 2 onboard PCIe switch (upstream)
          LnkCap Gen5 x16 · LnkSta Gen4 x4 (downgraded)
          ├─ 09:01.0 → 0a:00.0  Arc Pro B70 GPU die
          └─ 09:02.0 → 0b:00.0  HDMI audio
```

Notes:

- Each B70 card presents an onboard Intel PCIe switch (`8086:e2ff`) with the
  GPU die and its audio function on separate downstream ports.
- **BIOS bifurcation (x8/x8) is enabled** on the CPU x16 slot for a future
  expansion, but the current wiring is exactly what the tree shows: card 1 gets
  the full Gen4 x16 CPU link; card 2 is on the chipset's Gen4 x4 port. The
  bandwidth asymmetry doesn't hurt this workload — weights are loaded once and
  stay resident; steady-state inference traffic is small.
- `lspci` may report the GPU die links as `2.5GT/s x1` — that's the ASPM idle
  state, not the trained link. The card-level (switch upstream) links above are
  the meaningful numbers.

## Software

| Component | Version |
|---|---|
| OS | Ubuntu 26.04 LTS |
| Kernel | 7.0.0-28-generic |
| GPU kernel driver | Intel `xe` (Xe2 Graphics), in-kernel, vermagic 7.0.0-28-generic |
| GPU firmware | GuC 70.58.0 (`xe/bmg_guc_70.bin`), HuC 8.2.10 (`xe/bmg_huc.bin`) — from linux-firmware-intel-graphics 20260319.git217ca6e4-0ubuntu2.1 |
| GPU compute driver | Level Zero: libze1 1.28.6, libze-intel-gpu1 26.22.38646.6 |
| Intel oneAPI | 2026.0 — DPC++/C++ Compiler 2026.0.0 (2026.0.0.20260331), incl. oneMKL |
| cmake | 4.2.3 |
| gcc (host, not used for SYCL build) | 15.2.0 |
| llama.cpp | upstream master @ `876a4321163249c43ca4e986818fab5ab081f282` (submodule, unmodified) |

`LD_LIBRARY_PATH` at both build and runtime:
`/opt/intel/oneapi/compiler/2026.0/lib:/opt/intel/oneapi/2026.0/lib`
(both are required — libsvml/libdnnl/libiomp5 live in the first, libumf in the
second; missing libumf makes SYCL silently find zero platforms).

## Verified as-built configuration

The record of the actual build is `vendor/llama.cpp/build/CMakeCache.txt`
after running `scripts/build.sh`. Excerpt from the machine that produced the
benchmark (this is the build log's configure record — `scripts/build.sh`
reproduces it exactly):

```
GGML_SYCL:BOOL=ON
GGML_SYCL_F16:BOOL=ON
CMAKE_BUILD_TYPE:STRING=Release
CMAKE_C_COMPILER=/opt/intel/oneapi/compiler/2026.0/bin/icx
CMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2026.0/bin/icpx
CMAKE_CXX_FLAGS:STRING=-O0 -DGGML_SCHED_MAX_SPLIT_INPUTS=256
MKL_DIR=/opt/intel/oneapi/2026.0/lib/cmake/mkl
CMAKE_EXE_LINKER_FLAGS:STRING=-L/opt/intel/oneapi/compiler/2026.0/lib -liomp5 -lsvml -lirng -limf -lintlc
CMAKE_SHARED_LINKER_FLAGS:STRING=-L/opt/intel/oneapi/compiler/2026.0/lib -liomp5 -lsvml -lirng -limf -lintlc
```

Build time: ~30–40 min at `-j20` on the i7-12700KF. The resulting
`llama-server` reports version `b10216` (commit 876a43211).
