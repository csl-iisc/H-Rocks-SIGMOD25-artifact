#!/usr/bin/env bash
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$DIR/bin"
make -C "$DIR" bin/test_puts
OUT_DIR=output_puts
mkdir -p "$OUT_DIR"
for N in 10000 1000000 10000000 25000000 50000000 75000000 100000000; do
  DB="/pmem/hrocks_puts_${N}"
  rm -rf "$DB"
  echo "==> puts N=$N  (DB=$DB)"
  ./bin/test_puts -n "$N" -k 8 -v 8 -t 32 -f "$DB" > "$OUT_DIR/puts_${N}.log"
done
