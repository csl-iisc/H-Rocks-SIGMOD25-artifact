#!/usr/bin/env bash
set -euo pipefail

# Override via env if you like:
SIZES="${SIZES:-500 1000 2000 4000 8000 10000 16000 25000 50000 100000 200000 400000 800000 1000000 2000000 4000000 8000000 10000000 20000000 40000000 50000000 80000000 100000000}"
VAL_SIZE="${VAL_SIZE:-8}"
THREADS="${THREADS:-128}"
PREFILL_SIZE="${PREFILL_SIZE:-50000000}"
NQUERIES="${NQUERIES:-100}"
OUT_DIR="${OUT_DIR:-output_range}"
DB_PATH="/pmem/plush_table"

mkdir -p "$OUT_DIR" build
pushd ../build >/dev/null
cmake .. >/dev/null
make -j"$(nproc)" test_range
popd >/dev/null

for size in $SIZES; do
  echo "RANGE  arrival_n=$size  prefill=$PREFILL_SIZE  v=$VAL_SIZE  t=$THREADS"
  rm -rf "$DB_PATH" /dev/shm/*
  mkdir -p "$DB_PATH"
  ./build/test_range \
    -p "$PREFILL_SIZE" \
    -n "$size" \
    -e "$NQUERIES" \
    -t "$THREADS" \
    > "$OUT_DIR/output_${size}_k8_v${VAL_SIZE}_t${THREADS}"
done
