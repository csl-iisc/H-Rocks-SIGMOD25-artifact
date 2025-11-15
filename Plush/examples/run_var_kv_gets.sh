#!/usr/bin/env bash
set -euo pipefail

# Defaults (override via env: VAL_SIZES, KEY_SIZES, NKEYS, THREADS, OUT_DIR)
VAL_SIZES="${VAL_SIZES:-8 16 32 64 128 256 512 1024}"
KEY_SIZES="${KEY_SIZES:-8 16 32 64 128}"
NKEYS="${NKEYS:-5000000}"
THREADS="${THREADS:-32}"
OUT_DIR="${OUT_DIR:-output_var_kv_gets}"

mkdir -p "$OUT_DIR"
mkdir -p build
pushd build >/dev/null
cmake ../.. >/dev/null
make -j"$(nproc)" test_get_mt
popd >/dev/null

for keys in $NKEYS; do
  for key_size in $KEY_SIZES; do
    for val_size in $VAL_SIZES; do
      rm -rf /pmem/* /dev/shm/*
      mkdir -p /pmem/plush_table
      echo "GETS  n=$keys  k=$key_size  v=$val_size  t=$THREADS"
      ./build/test_get_mt -n "$keys" -k "$key_size" -v "$val_size" -t "$THREADS" \
        > "$OUT_DIR/output_${keys}_${key_size}_${val_size}"
    done
  done
done
