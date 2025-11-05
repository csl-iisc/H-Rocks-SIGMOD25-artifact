#!/usr/bin/env bash
# pmem_parse_deletes.sh
# Usage:
#   ./pmem_parse_deletes.sh [INPUT_DIR] [OUTPUT_CSV]
# Example:
#   ./pmem_parse_deletes.sh output_deletes pmem_deletes_throughput.csv
set -euo pipefail
IN_DIR="${1:-output_deletes}"
OUT_CSV="${2:-pmem_deletes_throughput.csv}"
SIZES_LIST="${SIZES:-}"

echo "size,throughput_ops_per_s" > "$OUT_CSV"
shopt -s nullglob

ms_from_key() {
  awk -v key="$1" '$0 ~ key { if (match($0,/([0-9]+(\.[0-9]+)?) ms/,a)) { print a[1]; exit } }' "$2"
}

for f in "$IN_DIR"/deletes_*.log "$IN_DIR"/delete_*.log; do
  [[ -e "$f" ]] || continue
  base="$(basename "$f")"                   # deletes_100000.log
  size="${base#*_}"; size="${size%.log}"    # 100000
  if [[ -n "$SIZES_LIST" && " $SIZES_LIST " != *" $size "* ]]; then continue; fi

  ms="$(ms_from_key 'delete_time' "$f")"; : "${ms:=0}"
  thr="$(awk -v n="$size" -v t="$ms" 'BEGIN{ if(t>0){val=(n*1000.0)/t; if(val>n) val=n; printf "%.2f",val} else printf "0"}')"
  echo "$size,$thr" >> "$OUT_CSV"
done

tmp="$(mktemp)"
sort -t, -k1,1n "$OUT_CSV" > "$tmp" && mv "$tmp" "$OUT_CSV"
echo "Wrote $(realpath "$OUT_CSV")"
