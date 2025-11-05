#!/usr/bin/env bash
# parse_updates.sh
# Usage: ./parse_updates.sh [INPUT_DIR] [OUTPUT_CSV]
# Ex:    ./parse_updates.sh output_updates updates_throughput.csv
set -euo pipefail
IN_DIR="${1:-.}"
OUT_CSV="${2:-updates_throughput.csv}"
SIZES_LIST="${SIZES:-}"   # optional: "100000 200000 ..."

echo "size,throughput_ops_per_s" > "$OUT_CSV"
shopt -s nullglob

get_num() { local k="$1" f="$2" v; v="$(grep -m1 -E "$k" "$f" 2>/dev/null | grep -Eo '[0-9]+(\.[0-9]+)?' | tail -1 || true)"; [[ -n "${v:-}" ]] && echo "$v" || echo 0; }

for f in "$IN_DIR"/output_8_8_*; do
  base="$(basename "$f")"
  size="${base##*_}"
  if [[ -n "$SIZES_LIST" ]]; then ok=false; for s in $SIZES_LIST; do [[ "$size" == "$s" ]] && ok=true; done; $ok || continue; fi

  s="$(get_num 'update_setup_time|setup_time' "$f")"
  k="$(get_num 'update_kernel_time|kernel_time' "$f")"
  so="$(get_num 'update_sort_time|sort_time' "$f")"
  m="$(get_num 'update_memtable_time|memtable_time' "$f")"

  total_ms="$(awk -v a="$s" -v b="$k" -v c="$so" -v d="$m" 'BEGIN{printf "%.6f",(a+b+c+d)}')"
  thr="$(awk -v n="$size" -v ms="$total_ms" 'BEGIN{if(ms>0){val=(n*1000.0)/ms;if(val>n)val=n;printf "%.2f",val}else printf "0"}')"
  echo "$size,$thr" >> "$OUT_CSV"
done

tmp="$(mktemp)"; sort -t, -k1,1n "$OUT_CSV" > "$tmp" && mv "$tmp" "$OUT_CSV"
echo "Wrote $(realpath "$OUT_CSV")"
