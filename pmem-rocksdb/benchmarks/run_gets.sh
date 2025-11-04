#!/usr/bin/env bash
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
make -C "$DIR" bin/test_gets
OUT_DIR=output_gets
mkdir -p "$OUT_DIR"
for N in 10000 1000000 10000000 25000000 50000000 75000000 100000000; do
  DB="/pmem/hrocks_gets_${N}"
  rm -rf "$DB"
  echo "==> gets N=$N (prefill=$N, DB=$DB)"
  ./bin/test_gets -p "$N" -g "$N" -k 8 -v 8 -t 32 -f "$DB" > "$OUT_DIR/gets_${N}.log"
done
