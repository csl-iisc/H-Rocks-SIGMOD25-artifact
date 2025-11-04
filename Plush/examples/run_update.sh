#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
BUILD_DIR="$ROOT/build"
BIN="$BUILD_DIR/test_update"
OUT_DIR="$ROOT/output_updates"

SIZES="10000 1000000 10000000 25000000 50000000 75000000 100000000"
THREADS="${THREADS:-32}"
PREFILL_SIZE="${PREFILL_SIZE:-50000000}"
JOBS="${JOBS:-10}"

mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR"/output_*
cmake -S "$ROOT" -B "$BUILD_DIR" >/dev/null
cmake --build "$BUILD_DIR" --target test_update -- -j "$JOBS" >/dev/null

for size in $SIZES; do
    echo "$size"
    rm -rf /pmem/plush_table
    mkdir -p /pmem/plush_table
    "$BIN" -p "$PREFILL_SIZE" -u "$size" -t "$THREADS" > "$OUT_DIR/output_8_8_$size"
done
