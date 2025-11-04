#!/usr/bin/env bash
# Parse uTree insert outputs -> CSV with ops/sec
# Usage: ./parse_inserts.sh [INPUT_DIR] [OUTPUT_CSV]
# Defaults: INPUT_DIR=output_inserts  OUTPUT_CSV=utree_inserts.csv
set -euo pipefail

IN_DIR="${1:-output_inserts}"
OUT_CSV="${2:-utree_inserts.csv}"
SIZES_LIST="${SIZES:-}"

echo "size,throughput_ops_per_s" > "$OUT_CSV"
shopt -s nullglob

for f in "$IN_DIR"/output_*; do
  base="$(basename "$f")"
  size="${base#output_}"
  size="${size%%_*}" # trim suffix (e.g., _8_8)

  if [[ -n "$SIZES_LIST" && " $SIZES_LIST " != *" $size "* ]]; then
    continue
  fi

  line="$(grep -E 'throughput:' "$f" | head -n1 || true)"
  [[ -z "$line" ]] && continue
  mops="$(awk '{print $(NF-1)}' <<<"$line")"
  ops="$(awk -v m="$mops" 'BEGIN{printf "%.6f", m*1e6}')"
  echo "$size,$ops" >> "$OUT_CSV"
done

tmp="$(mktemp)"
sort -t, -k1,1n "$OUT_CSV" > "$tmp" && mv "$tmp" "$OUT_CSV"
echo "Wrote $(realpath "$OUT_CSV")"
