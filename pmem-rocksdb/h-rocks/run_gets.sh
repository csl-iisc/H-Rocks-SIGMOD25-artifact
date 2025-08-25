#!/usr/bin/env bash
set -euo pipefail

SIZES="10000 1000000 10000000 25000000 50000000 75000000 100000000"
VAL_SIZES="8"
KEY_SIZE=8
OUT_DIR="output_get_put"
SLEEP_BETWEEN=2

mkdir -p "$OUT_DIR"
make lib
make bin/test_puts bin/test_get_put

for v in $VAL_SIZES; do
  for n in $SIZES; do
    echo "==> GET+PUT n=$n k=$KEY_SIZE v=$v (prefill first)"
    rm -rf /pmem/rocksdb_* /pmem/values* /dev/shm/*
    ./bin/test_puts    -n "$n" -k "$KEY_SIZE" -v "$v" > /dev/null
    sleep 1
    ./bin/test_get_put -n "$n" -k "$KEY_SIZE" -v "$v" > "$OUT_DIR/get_put_k${KEY_SIZE}_v${v}_n${n}.log"
    sleep "$SLEEP_BETWEEN"
  done
done
