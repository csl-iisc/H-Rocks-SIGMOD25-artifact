#!/usr/bin/env bash
# viper_parse_ycsbC.sh
set -euo pipefail
IN_DIR="${1:-output_ycsbC}"
OUT_CSV="${2:-viper_ycsbC_throughput.csv}"
SIZES_LIST="${SIZES:-}"

if [[ ! -d "$IN_DIR" ]]; then
  for alt in "${IN_DIR/output_ycsbC/output_ycsbC4}" \
             output_ycsbC4; do
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

echo "size,throughput_ops_per_s" > "$OUT_CSV"; shopt -s nullglob
for f in "$IN_DIR"/output_8_8_*; do
  base="$(basename "$f")"; size="${base##*_}"
  if [[ -n "$SIZES_LIST" && " $SIZES_LIST " != *" $size "* ]]; then continue; fi
  ms_line=""
  if ms_found="$(grep -m1 -E 'get_time|final_get_time' "$f" 2>/dev/null)"; then
    ms_line="$ms_found"
  fi
  ms="$(awk '{print $NF}' <<< "$ms_line")"; : "${ms:=0}"
  thr="$(awk -v n="$size" -v t="${ms:-0}" 'BEGIN{ if(t>0){val=(n*1000.0)/t; if(val>n) val=n; printf "%.2f",val} else printf "0"}')"
  echo "$size,$thr" >> "$OUT_CSV"
done
tmp="$(mktemp)"; sort -t, -k1,1n "$OUT_CSV" > "$tmp" && mv "$tmp" "$OUT_CSV"
echo "Wrote $(realpath "$OUT_CSV")"
