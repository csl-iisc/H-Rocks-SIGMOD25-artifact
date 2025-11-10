#!/usr/bin/env bash
# parse_ycsbB.sh
# Usage: ./parse_ycsbB.sh [INPUT_DIR] [OUTPUT_CSV]
# Ex:    ./parse_ycsbB.sh output_ycsbB ycsbB_throughput.csv
set -euo pipefail
IN_DIR="${1:-.}"
OUT_CSV="${2:-ycsbB_throughput.csv}"
SIZES_LIST="${SIZES:-}"

echo "size,throughput_ops_per_s" > "$OUT_CSV"
shopt -s nullglob

get_num() {
  local k="$1" f="$2" v
  v="$(grep -m1 -E "$k" "$f" 2>/dev/null | grep -Eo '[0-9]+(\.[0-9]+)?' | tail -1 || true)"
  [[ -n "${v:-}" ]] && echo "$v" || echo 0
}

for f in "$IN_DIR"/output_*; do
  [[ -f "$f" ]] || continue
  base="$(basename "$f")"
  IFS=_ read -r tag size rest <<<"$base"
  [[ "$tag" == "output" && "$size" =~ ^[0-9]+$ ]] || continue
  [[ -z "$SIZES_LIST" || " $SIZES_LIST " == *" $size "* ]] || continue

  ws="$(get_num 'setup_time|update_setup_time' "$f")"
  wso="$(get_num 'sort_time|update_sort_time' "$f")"
  ww="$(get_num 'write_batch_time|update_write_batch_time' "$f")"
  wm="$(get_num 'memtable_time|update_memtable_time' "$f")"
  rs="$(get_num 'read_setup_time'  "$f")"
  rk="$(get_num 'read_kernel_time' "$f")"
  rb="$(get_num 'read_batch_time'  "$f")"
  rc="$(get_num 'copy_back_time'   "$f")"

  total_ms="$(awk -v a="$ws" -v b="$wso" -v c="$ww" -v d="$wm" -v e="$rs" -v g="$rk" -v h="$rb" -v i="$rc" \
                 'BEGIN{printf "%.6f",(a+b+c+d+e+g+h+i)}')"
  thr="$(awk -v n="$size" -v ms="$total_ms" 'BEGIN{if(ms>0){val=(n*1000.0)/ms;if(val>n)val=n;printf "%.2f",val}else printf "0"}')"
  echo "$size,$thr" >> "$OUT_CSV"
done

tmp="$(mktemp)"
sort -t, -k1,1n "$OUT_CSV" > "$tmp" && mv "$tmp" "$OUT_CSV"
echo "Wrote $(realpath "$OUT_CSV")"
