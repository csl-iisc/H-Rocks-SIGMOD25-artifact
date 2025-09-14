#!/usr/bin/env bash
# parse_ycsbC.sh
# Usage: ./parse_ycsbC.sh [INPUT_DIR] [OUTPUT_CSV]
set -euo pipefail
IN_DIR="${1:-.}"
OUT_CSV="${2:-ycsbC_throughput.csv}"
SIZES_LIST="${SIZES:-}"

echo "size,throughput_ops_per_s" > "$OUT_CSV"
shopt -s nullglob
get_num(){ local k="$1" f="$2" v; v="$(grep -m1 -E "$k" "$f" 2>/dev/null | grep -Eo '[0-9]+(\.[0-9]+)?' | tail -1 || true)"; [[ -n "${v:-}" ]]&&echo "$v"||echo 0; }

for f in "$IN_DIR"/output_*_8_8; do
  base="$(basename "$f")"; IFS=_ read -r tag size t1 t2 <<<"$base"; [[ "$tag"=="output" && "$t1"=="8" && "$t2"=="8" ]] || continue
  [[ -z "$SIZES_LIST" || " $SIZES_LIST " == *" $size "* ]] || continue
  rs="$(get_num 'read_setup_time' "$f")"; rk="$(get_num 'read_kernel_time' "$f")"; rb="$(get_num 'read_batch_time' "$f")"; rc="$(get_num 'copy_back_time' "$f")"
  total_ms="$(awk -v a="$rs" -v b="$rk" -v c="$rb" -v d="$rc" 'BEGIN{printf "%.6f",(a+b+c+d)}')"
  thr="$(awk -v n="$size" -v ms="$total_ms" 'BEGIN{if(ms>0)printf "%.2f",(n*1000.0)/ms; else printf "0"}')"
  echo "$size,$thr" >> "$OUT_CSV"
done
tmp="$(mktemp)"; sort -t, -k1,1n "$OUT_CSV" > "$tmp" && mv "$tmp" "$OUT_CSV"; echo "Wrote $(realpath "$OUT_CSV")"
