#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

# Arrival request rates to sweep (ops/sec equivalent in your driver)
SIZES="${SIZES:-500 1000 2000 4000 8000 10000 16000 25000 50000 100000 200000 400000 800000 1000000 2000000 4000000 8000000 10000000}"

# Workload params (override via env as needed)
PREFILL="${PREFILL:-50000000}"   # -p
K="${K:-8}"                      # -k
V="${V:-8}"                      # -v
OUT_DIR="${OUT_DIR:-output_range}"

mkdir -p "$OUT_DIR"

# Build once
make -j"$(nproc)" lib
make -j"$(nproc)" bin/test_range

for g in $SIZES; do
  hr_rm_rf "${HR_PMEM_DIR:?}/"* "${HR_SHM_DIR:?}/"*
  echo "H-Rocks RANGE  prefill=$PREFILL  g=$g  k=$K  v=$V"
  ./bin/test_range -p "$PREFILL" -g "$g" -k "$K" -v "$V" \
    > "$OUT_DIR/output_${PREFILL}_${g}"
done
