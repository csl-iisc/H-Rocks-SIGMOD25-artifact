#!/usr/bin/env bash
# viper_parse_updates.sh
# Usage:
#   ./viper_parse_updates.sh [INPUT_DIR] [OUTPUT_CSV]
# Example:
#   ./viper_parse_updates.sh output_updates viper_updates_throughput.csv
# Optional:
#   SIZES="100000 200000 400000" ./viper_parse_updates.sh output_updates viper_updates_throughput.csv

set -euo pipefail
IN_DIR="${1:-output_updates}"
OUT_CSV="${2:-viper_updates_throughput.csv}"
SIZES_LIST="${SIZES:-}"

if [[ ! -d "$IN_DIR" ]]; then
  for alt in "${IN_DIR/output_updates/output_prefill_update}" \
             output_prefill_update; do
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
  base="$(basename "$f")"
  size="${base##*_}"                  # assumes ..._<size>
  if [[ -n "$SIZES_LIST" && " $SIZES_LIST " != *" $size "* ]]; then
    continue
  fi

  ms_line=""
  if ms_found="$(grep -m1 -E 'update_time' "$f" 2>/dev/null)"; then
    ms_line="$ms_found"
  fi
  ms="$(awk '{print $NF}' <<< "$ms_line")"
  : "${ms:=0}"

  thr="$(awk -v n="$size" -v t="$ms" 'BEGIN { if (t > 0) {val=(n*1000.0)/t; if(val>n) val=n; printf "%.2f",val} else printf "0" }')"
  echo "$size,$thr" >> "$OUT_CSV"
done

# sort numerically by size
tmp="$(mktemp)"
sort -t, -k1,1n "$OUT_CSV" > "$tmp" && mv "$tmp" "$OUT_CSV"

echo "Wrote $(realpath "$OUT_CSV")"
