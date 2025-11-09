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

should_include_size() {
  local size="$1"
  if [[ -z "$SIZES_LIST" ]]; then return 0; fi
  for s in $SIZES_LIST; do
    if [[ "$size" == "$s" ]]; then return 0; fi
  done
  return 1
}

emit_row() {
  local file="$1" size="$2"
  if ! should_include_size "$size"; then return; fi
  local s so w m total_ms thr
  s="$(get_num 'delete_setup_time|setup_time' "$file")"
  so="$(get_num 'delete_sort_time|sort_time' "$file")"
  w="$(get_num 'delete_write_batch_time|write_batch_time' "$file")"
  m="$(get_num 'delete_memtable_time|memtable_time' "$file")"
  total_ms="$(awk -v a="$s" -v b="$so" -v c="$w" -v d="$m" 'BEGIN{printf "%.6f",(a+b+c+d)}')"
  thr="$(awk -v n="$size" -v ms="$total_ms" 'BEGIN{if(ms>0){val=(n*1000.0)/ms;if(val>n)val=n;printf "%.2f",val}else printf "0"}')"
  echo "$size,$thr" >> "$OUT_CSV"
}

for f in "$IN_DIR"/output_8_8_*; do
  base="$(basename "$f")"; size="${base##*_}"
  emit_row "$f" "$size"
done

for f in "$IN_DIR"/deletes_k*_v*_n*.log; do
  base="$(basename "$f")"
  if [[ "$base" =~ deletes_k([0-9]+)_v([0-9]+)_n([0-9]+)\.log$ ]]; then
    size="${BASH_REMATCH[3]}"
    emit_row "$f" "$size"
  fi
done

tmp="$(mktemp)"; sort -t, -k1,1n "$OUT_CSV" > "$tmp" && mv "$tmp" "$OUT_CSV"
echo "Wrote $(realpath "$OUT_CSV")"
