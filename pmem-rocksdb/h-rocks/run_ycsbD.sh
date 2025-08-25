#!/usr/bin/env bash
set -euo pipefail

SIZES="10000 1000000 10000000 25000000 50000000 75000000 100000000"
VAL_SIZES="100"
KEY_SIZE="16"

OUT_DIR="output_ycsbD"
mkdir -p "$OUT_DIR"

make lib && make bin/test_ycsbD
for v in $VAL_SIZES; do
  for n in $PUT_SIZES; do
    echo "$n"
    rm -rf /pmem/* /dev/shm/*
    ./bin/test_ycsbD -n "$n" -k "$KEY_SIZE" -v "$v" \
      > "${OUT_DIR}/output_${n}_${KEY_SIZE}_${v}"
    sleep 1
  done
done
