#!/usr/bin/env bash
set -e
OUT_DIR=output_ycsbC
mkdir -p "$OUT_DIR"
for N in 10000 1000000 10000000 25000000 50000000 75000000 100000000; do
  DB="/pmem/hrocks_ycsbC_${N}"
  rm -rf "$DB"
  echo "==> YCSB-C ops=$N (prefill=$N, DB=$DB)"
  ./bin/test_ycsbC -p "$N" -n "$N" -k 8 -v 8 -t 32 -f "$DB" > "$OUT_DIR/ycsbC_${N}.log"
done
