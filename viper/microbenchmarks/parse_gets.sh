#!/usr/bin/env bash
# viper_parse_gets.sh
# Usage:
#   ./viper_parse_gets.sh [INPUT_DIR] [OUTPUT_CSV]
# Example:
#   ./viper_parse_gets.sh output_gets viper_gets_throughput.csv
# Optional:
#   VAL_SIZES="8 100" SIZES="100000 200000" ./viper_parse_gets.sh output_gets viper_gets_throughput.csv

set -euo pipefail
IN_DIR="${1:-output_gets}"
OUT_CSV="${2:-viper_gets_throughput.csv}"
VAL_SIZES="${VAL_SIZES:-8 100}"
SIZES_LIST="${SIZES:-}"

# Fall back to the actual output directories used by run scripts if needed
if [[ ! -d "$IN_DIR" ]]; then
  for alt in "${IN_DIR/output_gets/output_prefill_get}" \
             "${IN_DIR/output_gets/output_prefill_get2}" \
             output_prefill_get \
             output_prefill_get2; do
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

echo "size,val_size,throughput_ops_per_s" > "$OUT_CSV"
shopt -s nullglob

for v in $VAL_SIZES; do
  for f in "$IN_DIR"/output_8_"$v"_*; do
    base="$(basename "$f")"          # output_8_<v>_<size>
    size="${base##*_}"
    if [[ -n "$SIZES_LIST" && " $SIZES_LIST " != *" $size "* ]]; then continue; fi

    ms_line=""
    if ms_found="$(grep -m1 -E 'final_get_time' "$f" 2>/dev/null)"; then
      ms_line="$ms_found"
    fi
    ms="$(awk '{print $NF}' <<< "$ms_line")"
    : "${ms:=0}"
    thr="$(awk -v n="$size" -v t="${ms:-0}" 'BEGIN{ if(t>0){val=(n*1000.0)/t; if(val>n) val=n; printf "%.2f",val} else printf "0"}')"
    echo "$size,$v,$thr" >> "$OUT_CSV"
  done
done

tmp="$(mktemp)"; sort -t, -k1,1n -k2,2n "$OUT_CSV" > "$tmp" && mv "$tmp" "$OUT_CSV"
echo "Wrote $(realpath "$OUT_CSV")"
