#!/usr/bin/env bash
set -e
OUT_DIR=output_ycsbD
mkdir -p "$OUT_DIR"
for N in 10000 1000000 10000000 25000000 50000000 75000000 100000000; do
  DB="/pmem/hrocks_ycsbD_${N}"
  rm -rf "$DB"
  echo "==> YCSB-D ops=$N (prefill=$N, DB=$DB)"
  ./bin/test_ycsbD -p "$N" -n "$N" -k 8 -v 8 -t 32 -f "$DB" > "$OUT_DIR/ycsbD_${N}.log"
done
echo "All tests completed."