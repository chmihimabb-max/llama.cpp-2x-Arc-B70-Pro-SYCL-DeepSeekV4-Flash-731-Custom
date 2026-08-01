#!/usr/bin/env bash
# System tuning for RAM-resident LLM serving (run once; see README for why).
set -euo pipefail

DISK="${DISK:-nvme1n1}"   # block device holding the model files

echo "== 1. Disable swap (permanent) =="
# Comment out swap entries in /etc/fstab, e.g.:
#   #/swap.img   none  swap  sw  0  0
#   #/swapfile   none  swap  sw  0  0
# then:
sudo swapoff -a
sudo systemctl daemon-reload
swapon --show || true

echo "== 2. Zero block readahead on the model disk (until reboot) =="
# Default 128KB readahead amplifies scattered mmap expert faults ~3x.
echo 0 | sudo tee "/sys/block/${DISK}/queue/read_ahead_kb"

echo "== 3. Optional: persist readahead across reboots (udev rule) =="
cat <<EOF
Create /etc/udev/rules.d/99-llm-readahead.rules:

  ACTION=="add|change", KERNEL=="${DISK}", ATTR{queue/read_ahead_kb}="0"

Then: sudo udevadm control --reload && sudo udevadm trigger
EOF

echo "Done. Verify with: free -h && cat /sys/block/${DISK}/queue/read_ahead_kb"
