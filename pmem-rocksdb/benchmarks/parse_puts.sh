#!/usr/bin/env bash
# pmem_parse_puts.sh
# Usage: ./pmem_parse_puts.sh [INPUT_DIR] [OUTPUT_CSV]
# Ex:    ./pmem_parse_puts.sh output_puts pmem_puts_throughput.csv
set -euo pipefail
IN_DIR="${1:-output_puts}"
OUT_CSV="${2:-pmem_puts_throughput.csv}"
SIZES_LIST="${SIZES:-}"

echo "size,throughput_ops_per_s" > "$OUT_CSV"; shopt -s nullglob

ms_from_key() {  # extract "<number> ms" from the line containing the key
  awk -v key="$1" '
    $0 ~ key { if (match($0,/([0-9]+(\.[0-9]+)?) ms/,a)) { print a[1]; exit } }
  ' "$2"
}

for f in "$IN_DIR"/puts_*.log; do
  base="$(basename "$f")"                    # puts_100000.log
  size="${base#puts_}"; size="${size%.log}"  # 100000
  if [[ -n "$SIZES_LIST" && " $SIZES_LIST " != *" $size "* ]]; then continue; fi

  ms="$(ms_from_key 'prefill_time' "$f")"; : "${ms:=0}"
  thr="$(awk -v n="$size" -v t="$ms" 'BEGIN{ if(t>0){val=(n*1000.0)/t; if(val>n) val=n; printf "%.2f",val} else printf "0"}')"
  echo "$size,$thr" >> "$OUT_CSV"
done

tmp="$(mktemp)"; sort -t, -k1,1n "$OUT_CSV" > "$tmp" && mv "$tmp" "$OUT_CSV"
echo "Wrote $(realpath "$OUT_CSV")"
