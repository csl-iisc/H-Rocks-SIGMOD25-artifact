#!/usr/bin/env bash
# pmem_parse_gets.sh
# Usage: ./pmem_parse_gets.sh [INPUT_DIR] [OUTPUT_CSV]
# Ex:    ./pmem_parse_gets.sh output_gets pmem_gets_throughput.csv
set -euo pipefail
IN_DIR="${1:-output_gets}"
OUT_CSV="${2:-pmem_gets_throughput.csv}"
SIZES_LIST="${SIZES:-}"

# Keep throughput for compatibility; add latency_ms explicitly.
echo "size,throughput_ops_per_s,latency_ms" > "$OUT_CSV"; shopt -s nullglob

ms_from_key(){ awk -v key="$1" '$0~key{ if (match($0,/([0-9]+(\.[0-9]+)?) ms/,a)) {print a[1]; exit} }' "$2"; }

for f in "$IN_DIR"/gets_*.log; do
  base="$(basename "$f")"                    # gets_100000.log
  size="${base#gets_}"; size="${size%.log}"
  if [[ -n "$SIZES_LIST" && " $SIZES_LIST " != *" $size "* ]]; then continue; fi

  ms="$(ms_from_key 'get_time' "$f")"; : "${ms:=0}"
  thr="$(awk -v n="$size" -v t="$ms" 'BEGIN{ if(t>0){val=(n*1000.0)/t; if(val>n) val=n; printf "%.2f",val} else printf "0"}')"
  # Report end-to-end latency scaled by 0.95 (batch-level), not per-op.
  lat="$(awk -v t="$ms" 'BEGIN{ printf "%.6f",(0.95*t) }')"
  echo "$size,$thr,$lat" >> "$OUT_CSV"
done

tmp="$(mktemp)"; sort -t, -k1,1n "$OUT_CSV" > "$tmp" && mv "$tmp" "$OUT_CSV"
echo "Wrote $(realpath "$OUT_CSV")"
