#!/usr/bin/env bash
set -euo pipefail

SIZES="10000 1000000 10000000 25000000 50000000 75000000 100000000"
VAL_SIZES="100"
KEY_SIZE="16"

OUT_DIR="output_ycsbA"
DB_PATH="/pmem/rdb_ycsbA"   # match the hardcoded path used by the program

mkdir -p "$OUT_DIR"

# Build once
make lib && make bin/test_ycsbA

for v in $VAL_SIZES; do
  for n in $SIZES; do
    echo "==> ycsbA: ops=$n k=$KEY_SIZE v=$v"
    rm -rf "$DB_PATH" /dev/shm/*
    ./bin/test_ycsbA -n "$n" -k "$KEY_SIZE" -v "$v" \
      > "${OUT_DIR}/output_${n}_${KEY_SIZE}_${v}"
    sleep 1
  done
done
