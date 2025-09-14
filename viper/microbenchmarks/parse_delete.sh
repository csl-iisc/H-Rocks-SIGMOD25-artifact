#!/usr/bin/env bash
# viper_parse_deletes.sh
# Usage:
#   ./viper_parse_deletes.sh [INPUT_DIR] [OUTPUT_CSV]
# Example:
#   ./viper_parse_deletes.sh output_deletes viper_deletes_throughput.csv
# Limit sizes:
#   SIZES="100000 200000 400000" ./viper_parse_deletes.sh output_deletes viper_deletes_throughput.csv

set -euo pipefail
IN_DIR="${1:-output_deletes}"
OUT_CSV="${2:-viper_deletes_throughput.csv}"
SIZES_LIST="${SIZES:-}"

echo "size,throughput_ops_per_s" > "$OUT_CSV"
shopt -s nullglob

for f in "$IN_DIR"/output_*; do
  base="$(basename "$f")"          # e.g., output_8_8_100000 or output_100000
  size="${base##*_}"               # take last underscore field as size

  # optional whitelist of sizes
  if [[ -n "$SIZES_LIST" && " $SIZES_LIST " != *" $size "* ]]; then
    continue
  fi

  ms="$(grep -m1 -E 'delete_time' "$f" 2>/dev/null | awk '{print $NF}')"
  : "${ms:=0}"

  thr="$(awk -v n="$size" -v t="$ms" 'BEGIN{ if(t>0) printf "%.2f",(n*1000.0)/t; else printf "0"}')"
  echo "$size,$thr" >> "$OUT_CSV"
done

# sort numerically by size
tmp="$(mktemp)"
sort -t, -k1,1n "$OUT_CSV" > "$tmp" && mv "$tmp" "$OUT_CSV"

echo "Wrote $(realpath "$OUT_CSV")"
