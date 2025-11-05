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

if [[ ! -d "$IN_DIR" ]]; then
  for alt in "${IN_DIR/output_deletes/output_prefill_delete}" \
             "${IN_DIR/output_deletes/output_prefill_delete2}" \
             output_prefill_delete \
             output_prefill_delete2; do
    if [[ "$alt" != "$IN_DIR" && -d "$alt" ]]; then
      IN_DIR="$alt"
      break
    fi
  done
fi

if [[ ! -d "$IN_DIR" ]]; then
  echo "Input directory '$IN_DIR' not found." >&2
  exit 1
fi

echo "size,throughput_ops_per_s" > "$OUT_CSV"
shopt -s nullglob

for f in "$IN_DIR"/output_*; do
  base="$(basename "$f")"          # e.g., output_8_8_100000 or output_100000
  size="${base##*_}"               # take last underscore field as size

  # optional whitelist of sizes
  if [[ -n "$SIZES_LIST" && " $SIZES_LIST " != *" $size "* ]]; then
    continue
  fi

  ms_line=""
  if ms_found="$(grep -m1 -E 'delete_time' "$f" 2>/dev/null)"; then
    ms_line="$ms_found"
  fi
  ms="$(awk '{print $NF}' <<< "$ms_line")"
  : "${ms:=0}"

  thr="$(awk -v n="$size" -v t="$ms" 'BEGIN{ if(t>0){val=(n*1000.0)/t; if(val>n) val=n; printf "%.2f",val} else printf "0"}')"
  echo "$size,$thr" >> "$OUT_CSV"
done

# sort numerically by size
tmp="$(mktemp)"
sort -t, -k1,1n "$OUT_CSV" > "$tmp" && mv "$tmp" "$OUT_CSV"

echo "Wrote $(realpath "$OUT_CSV")"
