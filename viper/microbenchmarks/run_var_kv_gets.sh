#!/usr/bin/env bash
set -euo pipefail

# Override via env if you like
VAL_SIZES="${VAL_SIZES:-8 16 32 64 100 128 256 512 1024}"
KEY_SIZES="${KEY_SIZES:-8 16 32 64 100}"
NKEYS="${NKEYS:-1000000 5000000}"     # number of GET ops
PREFILL_SIZE="${PREFILL_SIZE:-10000000}"
THREADS="${THREADS:-128}"
OUT_DIR="${OUT_DIR:-output_var_kv_gets}"

mkdir -p "$OUT_DIR" build
pushd build >/dev/null
cmake .. >/dev/null
make -j"$(nproc)" prefill_get
popd >/dev/null

for n in $NKEYS; do
  for k in $KEY_SIZES; do
    for v in $VAL_SIZES; do
      rm -rf /pmem/* /dev/shm/*
      echo "GETS  g=$n  prefill=$PREFILL_SIZE  k=$k  v=$v  t=$THREADS"
      ./build/prefill_get -p "$PREFILL_SIZE" -n "$n" -k "$k" -v "$v" -t "$THREADS" \
        > "$OUT_DIR/output_${n}_${k}_${v}"
    done
  done
done
