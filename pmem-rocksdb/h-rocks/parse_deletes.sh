#!/usr/bin/env bash
# parse_deletes.sh
# Usage: ./parse_deletes.sh [INPUT_DIR] [OUTPUT_CSV]
# Ex:    ./parse_deletes.sh output_deletes delete_throughput.csv
set -euo pipefail
IN_DIR="${1:-.}"
OUT_CSV="${2:-delete_throughput.csv}"
SIZES_LIST="${SIZES:-}"

echo "size,throughput_ops_per_s" > "$OUT_CSV"
shopt -s nullglob

get_num() { local k="$1" f="$2" v; v="$(grep -m1 -E "$k" "$f" 2>/dev/null | grep -Eo '[0-9]+(\.[0-9]+)?' | tail -1 || true)"; [[ -n "${v:-}" ]] && echo "$v" || echo 0; }

for f in "$IN_DIR"/output_8_8_*; do
  base="$(basename "$f")"; size="${base##*_}"
  if [[ -n "$SIZES_LIST" ]]; then ok=false; for s in $SIZES_LIST; do [[ "$size" == "$s" ]] && ok=true; done; $ok || continue; fi

  s="$(get_num 'delete_setup_time|setup_time' "$f")"
  so="$(get_num 'delete_sort_time|sort_time' "$f")"
  w="$(get_num 'delete_write_batch_time|write_batch_time' "$f")"
  m="$(get_num 'delete_memtable_time|memtable_time' "$f")"
  total_ms="$(awk -v a="$s" -v b="$so" -v c="$w" -v d="$m" 'BEGIN{printf "%.6f",(a+b+c+d)}')"
  thr="$(awk -v n="$size" -v ms="$total_ms" 'BEGIN{if(ms>0){val=(n*1000.0)/ms;if(val>n)val=n;printf "%.2f",val}else printf "0"}')"
  echo "$size,$thr" >> "$OUT_CSV"
done

tmp="$(mktemp)"; sort -t, -k1,1n "$OUT_CSV" > "$tmp" && mv "$tmp" "$OUT_CSV"
echo "Wrote $(realpath "$OUT_CSV")"
