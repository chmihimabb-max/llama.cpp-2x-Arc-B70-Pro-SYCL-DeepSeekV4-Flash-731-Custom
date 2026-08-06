#!/usr/bin/env bash
# System tuning for RAM-resident LLM serving (run once per boot; see README for why).
set -euo pipefail

DISK="${DISK:-nvme0n1}"   # block device holding the model files (was nvme1n1 while the model lived on the OS disk; moved 2026-08-01)

echo "== 1. Disable swap (permanent) =="
# Comment out swap entries in /etc/fstab, e.g.:
#   #/swap.img   none  swap  sw  0  0
#   #/swapfile   none  swap  sw  0  0
# then:
sudo swapoff -a
sudo systemctl daemon-reload
swapon --show || true

echo "== 2. Set block readahead on the model disk to 2 MiB (2048 KiB) =="
# DO NOT zero this. read_ahead_kb=0 makes every mmap page fault a lone 4 KiB
# read -> initial load runs latency-bound at ~240 MiB/s on a drive that does
# 1.75 GiB/s O_DIRECT. The old "0" recipe was a 512K SSD-streaming-regime
# trick (128 KB readahead amplified scattered expert faults ~3x); at 256K
# residency the only re-reads are the ~6 GiB/run page-cache rotation, and
# readahead makes those ~4-7 s of stalls instead of ~25 s.
sudo blockdev --setra 4096 "/dev/${DISK}"   # 4096 x 512B sectors = 2 MiB

echo "== 3. Persist readahead across reboots (udev rule) =="
# The rule below is created by this script (idempotent). sysfs value is KiB:
#   2048 KiB = 2 MiB.
RULE="/etc/udev/rules.d/99-llm-readahead.rules"
if [ ! -f "${RULE}" ]; then
    echo "ACTION==\"add|change\", KERNEL==\"${DISK}\", ATTR{queue/read_ahead_kb}=\"2048\"" | sudo tee "${RULE}"
else
    echo "${RULE} already exists, leaving as-is (verify it matches):"
    cat "${RULE}"
fi
sudo udevadm control --reload && sudo udevadm trigger

echo "Done. Verify with: free -h && cat /sys/block/${DISK}/queue/read_ahead_kb"
