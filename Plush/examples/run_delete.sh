#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
BUILD_DIR="$ROOT/build"
BIN="$BUILD_DIR/test_delete"
OUT_DIR="$ROOT/output_deletes"

SIZES="500 1000 2000 4000 8000 10000 16000 25000 50000 100000 200000 400000 800000 1000000 2000000 4000000 8000000 10000000 20000000 40000000 50000000 80000000 100000000"
VAL_SIZE="${VAL_SIZE:-8}"
THREADS="${THREADS:-32}"
PREFILL_SIZE="${PREFILL_SIZE:-50000000}"
JOBS="${JOBS:-10}"

mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR"/output_*
cmake -S "$ROOT" -B "$BUILD_DIR" >/dev/null
cmake --build "$BUILD_DIR" --target test_delete -- -j "$JOBS" >/dev/null

for size in $SIZES; do
    echo "$size"
    rm -rf /pmem/plush_table
    mkdir -p /pmem/plush_table
    "$BIN" -p "$PREFILL_SIZE" -n "$size" -k 8 -v "$VAL_SIZE" -t "$THREADS" > "$OUT_DIR/output_8_8_$size"
done
