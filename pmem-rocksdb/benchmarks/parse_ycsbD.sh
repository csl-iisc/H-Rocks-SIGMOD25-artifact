#!/usr/bin/env bash
# pmem_parse_ycsbD_min.sh
set -euo pipefail
IN_DIR="${1:-output_ycsbD}"
OUT_CSV="${2:-pmem_ycsbD_throughput.csv}"
SIZES_LIST="${SIZES:-}"

echo "size,throughput_ops_per_s" > "$OUT_CSV"; shopt -s nullglob
ms_from_key(){ awk -v key="$1" '$0~key{ if (match($0,/([0-9]+(\.[0-9]+)?) ms/,a)) {print a[1]; exit} }' "$2"; }

for f in "$IN_DIR"/ycsbD_*.log; do
  base="$(basename "$f")"; size="${base#ycsbD_}"; size="${size%.log}"
  if [[ -n "$SIZES_LIST" && " $SIZES_LIST " != *" $size "* ]]; then continue; fi
  p="$(ms_from_key 'put_time' "$f")"; : "${p:=0}"
  g="$(ms_from_key 'get_time' "$f")"; : "${g:=0}"
  total_ms="$(awk -v a="$p" -v b="$g" 'BEGIN{printf "%.6f",(a+b)}')"
  thr="$(awk -v n="$size" -v t="$total_ms" 'BEGIN{ if(t>0) printf "%.2f",(n*1000.0)/t; else printf "0"}')"
  echo "$size,$thr" >> "$OUT_CSV"
done

tmp="$(mktemp)"; sort -t, -k1,1n "$OUT_CSV" > "$tmp" && mv "$tmp" "$OUT_CSV"; echo "Wrote $(realpath "$OUT_CSV")"
