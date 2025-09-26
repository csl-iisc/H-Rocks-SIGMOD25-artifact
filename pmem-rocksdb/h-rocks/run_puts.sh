#!/usr/bin/env bash
set -euo pipefail

SIZES="10000 1000000 10000000 25000000 50000000 75000000 100000000"
VAL_SIZES="8"
KEY_SIZE=8
OUT_DIR="output_puts"
SLEEP_BETWEEN=2

mkdir -p "$OUT_DIR"
make clean && make lib
make bin/test_puts

for v in $VAL_SIZES; do
  for n in $SIZES; do
    echo "==> PUTS n=$n k=$KEY_SIZE v=$v"
    rm -rf /pmem/rocksdb_* /pmem/values* /dev/shm/rocksdb_*
    ./bin/test_puts -n "$n" -k "$KEY_SIZE" -v "$v" > "$OUT_DIR/puts_k${KEY_SIZE}_v${v}_n${n}.log"
    sleep "$SLEEP_BETWEEN"
  done
done
