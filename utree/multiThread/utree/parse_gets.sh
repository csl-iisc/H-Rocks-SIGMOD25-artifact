#!/usr/bin/env bash
# Parse uTree get outputs -> CSV with ops/sec
# Usage: ./parse_gets.sh [INPUT_DIR] [OUTPUT_CSV]
# Defaults: INPUT_DIR=output_gets  OUTPUT_CSV=utree_gets.csv
set -euo pipefail

IN_DIR="${1:-output_gets}"
OUT_CSV="${2:-utree_gets.csv}"
SIZES_LIST="${SIZES:-}"

echo "size,throughput_ops_per_s" > "$OUT_CSV"
shopt -s nullglob

for f in "$IN_DIR"/output_*; do
  base="$(basename "$f")"
  size="${base#output_}"
  size="${size%%_*}"

  if [[ -n "$SIZES_LIST" && " $SIZES_LIST " != *" $size "* ]]; then
    continue
  fi

  line="$(grep -E 'throughput:' "$f" | tail -n1 || true)"
  [[ -z "$line" ]] && continue
  mops="$(awk '{print $(NF-1)}' <<<"$line")"
  ops="$(awk -v m="$mops" -v n="$size" 'BEGIN{val=m*1e6;if(val>n)val=n;printf "%.6f",val}')"
  echo "$size,$ops" >> "$OUT_CSV"
done

tmp="$(mktemp)"
sort -t, -k1,1n "$OUT_CSV" > "$tmp" && mv "$tmp" "$OUT_CSV"
echo "Wrote $(realpath "$OUT_CSV")"
