#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve a writable pmem-like path for the DB.
DB_ROOT="${DB_ROOT:-}"
if [[ -z "$DB_ROOT" ]]; then
  if [[ -d /pmem && -w /pmem ]]; then
    DB_ROOT="/pmem"
  else
    DB_ROOT="$SCRIPT_DIR/tmp/pmem"
  fi
fi
mkdir -p "$DB_ROOT"

PREFILL_SIZE="${PREFILL_SIZE:-10000000}"
SIZES="${SIZES:-10000 50000 100000 500000 1000000 5000000 10000000}"
KEY_SIZE="${K:-8}"
VAL_SIZE="${V:-8}"
OUT_DIR="${OUT_DIR:-output_atomic_txn}"

mkdir -p "$OUT_DIR"
make -C "$SCRIPT_DIR" -j"$(nproc)" bin/test_atomic_txn

for size in $SIZES; do
  db_path="$DB_ROOT/rocksdb_atomic_txn_${size}"
  rm -rf "$db_path"
  echo "RocksDB atomic txn: n=$PREFILL_SIZE b=$size k=$KEY_SIZE v=$VAL_SIZE"
  /usr/bin/time -f "elapsed_s %e" "$SCRIPT_DIR/bin/test_atomic_txn" \
    -n "$PREFILL_SIZE" -b "$size" -k "$KEY_SIZE" -v "$VAL_SIZE" -f "$db_path" \
    > "$OUT_DIR/output_${PREFILL_SIZE}_${size}" 2>&1
done
