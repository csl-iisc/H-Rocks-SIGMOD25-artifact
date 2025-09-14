#!/usr/bin/env bash
set -euo pipefail

# Arrival request rates to sweep (ops = number of ranges)
ARRIVALS="${ARRIVALS:-10000 25000 50000 100000 1000000 5000000 10000000}"

# Workload params (override via env if needed)
KEY="${KEY:-8}"
VAL="${VAL:-8}"
THREADS="${THREADS:-32}"
PREFILL="${PREFILL:-50000000}"
RANGE_LEN="${RANGE_LEN:-100}"     # -L keys per range
RANDOMIZE="${RANDOMIZE:-1}"       # 1 => pass -r
OUT_DIR="${OUT_DIR:-output_range}"

# Build (expects Makefile target bin/test_range)
make lib
make -j"$(nproc)" bin/test_range

mkdir -p "$OUT_DIR"

for N in $ARRIVALS; do
  DB="/pmem/pmem_range_${N}"
  rm -rf "$DB" /dev/shm/*
  echo "==> range N=$N (prefill=$PREFILL, L=$RANGE_LEN, k=$KEY, v=$VAL, t=$THREADS, rand=$RANDOMIZE)"
  cmd=(./bin/test_range -p "$PREFILL" -q "$N" -L "$RANGE_LEN" -k "$KEY" -v "$VAL" -t "$THREADS" -f "$DB")
  [[ "$RANDOMIZE" == "1" ]] && cmd+=(-r)
  "${cmd[@]}" > "$OUT_DIR/range_${N}.log"
done
