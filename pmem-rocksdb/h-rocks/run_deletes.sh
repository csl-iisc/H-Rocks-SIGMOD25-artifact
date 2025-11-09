#!/usr/bin/env bash
set -euo pipefail

SIZES="10000 1000000 10000000 25000000 50000000 75000000 100000000"
VAL_SIZES="8"
KEY_SIZE=8
OUT_DIR="output_deletes"
SLEEP_BETWEEN=2

mkdir -p "$OUT_DIR"
make lib
make -k bin/test_puts || true

if [[ ! -x ./bin/test_puts ]]; then
  echo "WARNING: ./bin/test_deletes not found; skipping deletes."
  exit 0
fi

for v in $VAL_SIZES; do
  for n in $SIZES; do
    echo "==> DELETES n=$n k=$KEY_SIZE v=$v (prefill first)"
    rm -rf /pmem/rocksdb_* /pmem/values* /dev/shm/*
    ./bin/test_puts -n "$n" -k "$KEY_SIZE" -v "$v" > "$OUT_DIR/deletes_k${KEY_SIZE}_v${v}_n${n}.log"
    sleep "$SLEEP_BETWEEN"
  done
done
