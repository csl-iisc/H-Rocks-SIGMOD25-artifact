#!/usr/bin/env bash
# parse_gets_min.sh
# Usage:
#   ./parse_gets_min.sh [INPUT_DIR] [OUTPUT_CSV]
# Example:
#   ./parse_gets_min.sh . get_throughput.csv
#   ./parse_gets_min.sh output_gets get_throughput.csv
#
# Notes:
# - Looks for files named like: output_50M_8_8_<size>
# - If $SIZES is set (space-separated), only those sizes are included.

set -euo pipefail

IN_DIR="${1:-.}"
OUT_CSV="${2:-get_throughput.csv}"
SIZES_LIST="${SIZES:-}"   # optional whitelist, e.g. "100000 200000 400000"

echo "size,throughput_ops_per_s" > "$OUT_CSV"
shopt -s nullglob

# Extract last numeric (float) on the matching line; 0 if not found
get_num() {
  local key="$1" file="$2"
  local v
  v="$(grep -m1 -E "$key" "$file" 2>/dev/null | grep -Eo '[0-9]+(\.[0-9]+)?' | tail -1 || true)"
  [[ -n "${v:-}" ]] && echo "$v" || echo "0"
}

for f in "$IN_DIR"/output_50M_*_*_*; do
  base="$(basename "$f")"         # e.g., output_50M_8_8_100000
  size="${base##*_}"              # -> 100000

  # optional whitelist filter
  if [[ -n "$SIZES_LIST" ]]; then
    include="false"
    for s in $SIZES_LIST; do
      if [[ "$size" == "$s" ]]; then include="true"; break; fi
    done
    [[ "$include" == "true" ]] || continue
  fi

  rs="$(get_num 'read_setup_time'  "$f")"
  rk="$(get_num 'read_kernel_time' "$f")"
  rb="$(get_num 'read_batch_time'  "$f")"
  cb="$(get_num 'copy_back_time'   "$f")"

  total_ms="$(awk -v a="$rs" -v b="$rk" -v c="$rb" -v d="$cb" 'BEGIN{printf "%.6f",(a+b+c+d)}')"
  thr_ops_s="$(awk -v n="$size" -v ms="$total_ms" 'BEGIN{if(ms>0)printf "%.2f",(n*1000.0)/ms;else printf "0"}')"

  echo "${size},${thr_ops_s}" >> "$OUT_CSV"
done

# sort numerically by size
tmp="$(mktemp)"
sort -t, -k1,1n "$OUT_CSV" > "$tmp" && mv "$tmp" "$OUT_CSV"

echo "Wrote $(realpath "$OUT_CSV")"
