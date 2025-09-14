#!/usr/bin/env bash
# viper_parse_puts.sh
# Usage:
#   ./viper_parse_puts.sh [INPUT_DIR] [OUTPUT_CSV]
# Example:
#   ./viper_parse_puts.sh output_puts viper_puts_throughput.csv
# Optional:
#   SIZES="100000 200000 400000" ./viper_parse_puts.sh output_puts viper_puts_throughput.csv

set -euo pipefail
IN_DIR="${1:-output_puts}"
OUT_CSV="${2:-viper_puts_throughput.csv}"
SIZES_LIST="${SIZES:-}"

echo "size,throughput_ops_per_s" > "$OUT_CSV"
shopt -s nullglob

for f in "$IN_DIR"/output_8_8_*; do
  base="$(basename "$f")"           # output_8_100_<size>
  size="${base##*_}"
  if [[ -n "$SIZES_LIST" && " $SIZES_LIST " != *" $size "* ]]; then continue; fi

  ms="$(grep -m1 -E 'prefill_time' "$f" 2>/dev/null | awk '{print $NF}')"
  : "${ms:=0}"
  thr="$(awk -v n="$size" -v t="${ms:-0}" 'BEGIN{ if(t>0) printf "%.2f", (n*1000.0)/t; else printf "0"}')"
  echo "$size,$thr" >> "$OUT_CSV"
done

tmp="$(mktemp)"; sort -t, -k1,1n "$OUT_CSV" > "$tmp" && mv "$tmp" "$OUT_CSV"
echo "Wrote $(realpath "$OUT_CSV")"
