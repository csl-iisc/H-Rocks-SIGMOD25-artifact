#!/usr/bin/env bash
set -euo pipefail

# Sizes & params
SIZES="10000 1000000 10000000 25000000 50000000 75000000 100000000"
VAL_SIZES="8"
PREFILL_SIZE="50000000"
KEY_SIZE="8"

OUT_DIR="output_updates"
DB_PATH="/pmem/rocksdb_updates"

mkdir -p "$OUT_DIR"

# Build once
make clean && make lib
make bin/test_updates

for v in $VAL_SIZES; do
  for n in $SIZES; do
    echo "==> updates: prefill=$PREFILL_SIZE ops=$n k=$KEY_SIZE v=$v"
    rm -rf "$DB_PATH" "${DB_PATH}_"* /pmem/rocksdb_* /pmem/values* /pmem/hrocks_* \
      /dev/shm/rocksdb_* /dev/shm/values* /dev/shm/hrocks_* 2>/dev/null || true
    ./bin/test_updates -p "$PREFILL_SIZE" -n "$n" -k "$KEY_SIZE" -v "$v" \
      > "${OUT_DIR}/output_${KEY_SIZE}_${v}_${n}"
    sleep 2
  done
done
