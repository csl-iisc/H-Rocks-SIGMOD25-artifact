#!/usr/bin/env bash
# viper_parse_ycsbA.sh
# Usage:
#   ./viper_parse_ycsbA.sh [INPUT_DIR] [OUTPUT_CSV]
# Example:
#   ./viper_parse_ycsbA.sh output_ycsbA viper_ycsbA_throughput.csv
# Optional:
#   SIZES="100000 200000" ./viper_parse_ycsbA.sh output_ycsbA viper_ycsbA_throughput.csv

set -euo pipefail
IN_DIR="${1:-output_ycsbA}"
OUT_CSV="${2:-viper_ycsbA_throughput.csv}"
SIZES_LIST="${SIZES:-}"

if [[ ! -d "$IN_DIR" ]]; then
  for alt in "${IN_DIR/output_ycsbA/output_ycsbA4}" \
             output_ycsbA; do
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

  pt_line=""
  if pt_found="$(grep -m1 -E 'put_time' "$f" 2>/dev/null)"; then
    pt_line="$pt_found"
  fi
  gt_line=""
  if gt_found="$(grep -m1 -E 'get_time' "$f" 2>/dev/null)"; then
    gt_line="$gt_found"
  fi
  pt="$(awk '{print $NF}' <<< "$pt_line")"; : "${pt:=0}"
  gt="$(awk '{print $NF}' <<< "$gt_line")"; : "${gt:=0}"
  total_ms="$(awk -v a="$pt" -v b="$gt" 'BEGIN{printf "%.6f",(a+b)}')"
  thr="$(awk -v n="$size" -v t="$total_ms" 'BEGIN{ if(t>0){val=(n*1000.0)/t; if(val>n) val=n; printf "%.2f",val} else printf "0"}')"
  echo "$size,$thr" >> "$OUT_CSV"
done

tmp="$(mktemp)"; sort -t, -k1,1n "$OUT_CSV" > "$tmp" && mv "$tmp" "$OUT_CSV"
echo "Wrote $(realpath "$OUT_CSV")"
