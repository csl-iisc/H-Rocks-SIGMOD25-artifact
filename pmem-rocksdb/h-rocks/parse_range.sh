#!/usr/bin/env bash
# parse_range_min.sh
# Usage: ./parse_range_min.sh [INPUT_DIR] [OUTPUT_CSV]
# Ex:    ./parse_range_min.sh output_range range_throughput.csv
set -euo pipefail
IN_DIR="${1:-.}"
OUT_CSV="${2:-range_throughput.csv}"
SIZES_LIST="${SIZES:-}"

echo "size,throughput_ops_per_s" > "$OUT_CSV"
shopt -s nullglob

get_num() { local k="$1" f="$2" v; v="$(grep -m1 -E "$k" "$f" 2>/dev/null | grep -Eo '[0-9]+(\.[0-9]+)?' | tail -1 || true)"; [[ -n "${v:-}" ]] && echo "$v" || echo 0; }

# match both output_50M_<size> and output_50M_*_*_<size>
for f in "$IN_DIR"/output_50M_* "$IN_DIR"/output_50M_*_*_*; do
  [[ -e "$f" ]] || continue
  base="$(basename "$f")"
  size="${base##*_}"
  if [[ -n "$SIZES_LIST" ]]; then ok=false; for s in $SIZES_LIST; do [[ "$size" == "$s" ]] && ok=true; done; $ok || continue; fi

  a="$(get_num 'range_setup_time'     "$f")"
  b="$(get_num 'sum_kernel_time'      "$f")"
  c="$(get_num 'search_memtable_time' "$f")"
  d="$(get_num 'memcpy_kernel_time'   "$f")"

  total_ms="$(awk -v a="$a" -v b="$b" -v c="$c" -v d="$d" 'BEGIN{printf "%.6f",(a+b+c+d)}')"
  thr="$(awk -v n="$size" -v ms="$total_ms" 'BEGIN{if(ms>0)printf "%.2f",(n*1000.0)/ms; else printf "0"}')"
  echo "$size,$thr" >> "$OUT_CSV"
done

tmp="$(mktemp)"; sort -t, -k1,1n "$OUT_CSV" > "$tmp" && mv "$tmp" "$OUT_CSV"
echo "Wrote $(realpath "$OUT_CSV")"
