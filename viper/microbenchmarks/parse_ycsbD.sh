#!/usr/bin/env bash
# viper_parse_ycsbD.sh
# Usage:
#   ./viper_parse_ycsbD.sh [INPUT_DIR] [OUTPUT_CSV]
# Example:
#   ./viper_parse_ycsbD.sh output_ycsbD viper_ycsbD_throughput.csv

set -euo pipefail
IN_DIR="${1:-output_ycsbD}"
OUT_CSV="${2:-viper_ycsbD_throughput.csv}"
SIZES_LIST="${SIZES:-}"

if [[ ! -d "$IN_DIR" ]]; then
  for alt in "${IN_DIR/output_ycsbD/output_ycsbD4}" \
             output_ycsbD4 \
             output_ycsbD; do
    if [[ "$alt" != "$IN_DIR" && -d "$alt" ]]; then
      IN_DIR="$alt"
      break
    fi
  done
fi

if [[ ! -d "$IN_DIR" ]]; then
  echo "Input directory '$IN_DIR' not found." >&2
  exit 1
fi

echo "size,throughput_ops_per_s" > "$OUT_CSV"
shopt -s nullglob

for f in "$IN_DIR"/output_*; do
  [[ -f "$f" ]] || continue
  base="$(basename "$f")"
  if [[ "$base" =~ ^output_[0-9]+_[0-9]+_([0-9]+)$ ]]; then
    size="${BASH_REMATCH[1]}"
  else
    continue
  fi
  if [[ -n "$SIZES_LIST" && " $SIZES_LIST " != *" $size "* ]]; then continue; fi

  pt_line="$(grep -m1 -E 'put_time' "$f" 2>/dev/null || true)"
  gt_line="$(grep -m1 -E 'get_time' "$f" 2>/dev/null || true)"
  pt="$(awk '{print $NF}' <<< "$pt_line")"; : "${pt:=0}"
  gt="$(awk '{print $NF}' <<< "$gt_line")"; : "${gt:=0}"
  total_ms="$(awk -v a="$pt" -v b="$gt" 'BEGIN{printf "%.6f",(a+b)}')"
  thr="$(awk -v n="$size" -v t="$total_ms" 'BEGIN{ if(t>0){val=(n*1000.0)/t; if(val>n) val=n; printf "%.2f",val} else printf "0"}')"
  echo "$size,$thr" >> "$OUT_CSV"
done

tmp="$(mktemp)"
sort -t, -k1,1n "$OUT_CSV" > "$tmp" && mv "$tmp" "$OUT_CSV"
echo "Wrote $(realpath "$OUT_CSV")"
