#!/usr/bin/env bash
# parse_puts.sh
# Usage:
#   ./parse_puts.sh [INPUT_DIR] [OUTPUT_CSV]
# Example:
#   ./parse_puts.sh output_puts put_throughput.csv
#   ./parse_puts.sh output_puts put_throughput.csv

set -euo pipefail

IN_DIR="${1:-output_puts}"
OUT_CSV="${2:-put_throughput.csv}"

echo "size,throughput_ops_per_s" > "$OUT_CSV"
shopt -s nullglob

# extract last numeric on a matching line from a file (or 0 if missing)
get_num() {
  local key="$1" file="$2"
  local v
  v="$(grep -m1 -E "$key" "$file" 2>/dev/null | grep -Eo '[0-9]+(\.[0-9]+)?' | tail -1 || true)"
  [[ -n "${v:-}" ]] && echo "$v" || echo "0"
}

for f in "$IN_DIR"/output_*_*_*; do
  base="$(basename "$f")"         # e.g., output_8_8_100000
  size="${base##*_}"              # -> 100000

  setup_ms="$(get_num 'setup_time'         "$f")"
  sort_ms="$(get_num 'sort_time'           "$f")"
  write_ms="$(get_num 'write_batch_time'   "$f")"
  memtbl_ms="$(get_num 'memtable_time'     "$f")"
  # sst_setup_time intentionally ignored

  total_ms="$(awk -v a="$setup_ms" -v b="$sort_ms" -v c="$write_ms" -v d="$memtbl_ms" \
                 'BEGIN{printf "%.6f",(a+b+c+d)}')"
  thr_ops_s="$(awk -v n="$size" -v ms="$total_ms" \
                 'BEGIN{if(ms>0)printf "%.2f",(n*1000.0)/ms;else printf "0"}')"

  echo "${size},${thr_ops_s}" >> "$OUT_CSV"
done

# sort by size numerically
tmp="$(mktemp)"
sort -t, -k1,1n "$OUT_CSV" > "$tmp" && mv "$tmp" "$OUT_CSV"

echo "Wrote $(realpath "$OUT_CSV")"
