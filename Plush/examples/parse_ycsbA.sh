#!/usr/bin/env bash
# plush_parse_ycsbA.sh
# Usage: ./plush_parse_ycsbA.sh [INPUT_DIR] [OUTPUT_CSV]
set -euo pipefail
IN_DIR="${1:-.}"
OUT_CSV="${2:-plush_ycsbA_throughput.csv}"
SIZES_LIST="${SIZES:-}"

echo "size,throughput_ops_per_s" > "$OUT_CSV"; shopt -s nullglob
for f in "$IN_DIR"/output_8_8_*; do
  base="$(basename "$f")"; size="${base##*_}"
  [[ -n "$SIZES_LIST" && " $SIZES_LIST " != *" $size "* ]] && continue
  pt="$(grep -m1 -E 'put_time' "$f" 2>/dev/null | awk '{print $NF}')"; : "${pt:=0}"
  gt="$(grep -m1 -E 'get_time' "$f" 2>/dev/null | awk '{print $NF}')"; : "${gt:=0}"
  total_ms="$(awk -v a="$pt" -v b="$gt" 'BEGIN{printf "%.6f",(a+b)}')"
  thr="$(awk -v n="$size" -v t="$total_ms" 'BEGIN{ if(t>0) printf "%.2f",(n*1000.0)/t; else printf "0"}')"
  echo "$size,$thr" >> "$OUT_CSV"
done
tmp="$(mktemp)"; sort -t, -k1,1n "$OUT_CSV" > "$tmp" && mv "$tmp" "$OUT_CSV"
echo "Wrote $(realpath "$OUT_CSV")"
