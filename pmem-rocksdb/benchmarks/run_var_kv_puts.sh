#!/usr/bin/env bash
set -e

OUT_DIR=output_var_kv_puts
mkdir -p "$OUT_DIR"

# tweak if you want fewer/more sizes
KEY_SIZES="8 16 32 64 128"
VAL_SIZES="8 16 32 64 128 256 512 1024"

for N in 10000 1000000 10000000 25000000 50000000 75000000 100000000; do
  for K in $KEY_SIZES; do
    for V in $VAL_SIZES; do
      DB="/pmem/rocksdb_puts_${N}_k${K}_v${V}"
      rm -rf "$DB"
      echo "==> puts N=$N k=$K v=$V (DB=$DB)"
      ./bin/test_puts -n "$N" -k "$K" -v "$V" -t 32 -f "$DB" > "$OUT_DIR/puts_${N}_${K}_${V}.log"
    done
  done
done
