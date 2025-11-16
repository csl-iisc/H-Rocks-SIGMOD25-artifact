#!/usr/bin/env bash
# Usage: ./parse_range.sh [INPUT_DIR] [OUTPUT_CSV]
# Default: INPUT_DIR=output_range  OUTPUT_CSV=plush_range_throughput.csv
# Optional filter: SIZES="10000 100000 1000000" ./parse_range.sh
set -euo pipefail

IN_DIR="${1:-output_range}"
OUT_CSV="${2:-plush_range_throughput.csv}"
SIZES_LIST="${SIZES:-}"

echo "size,throughput_ops_per_s" > "$OUT_CSV"
shopt -s nullglob

in_list(){ [[ -z "${2:-}" ]] && return 0; [[ " $2 " == *" $1 "* ]]; }

# extract "<number> ms" from the first line containing 'range_time'
ms_from_file(){ grep -m1 -E 'range_time' "$1" 2>/dev/null | awk '{print $NF}'; }

for f in "$IN_DIR"/output_*; do
  base="$(basename "$f")"            # output_<size>_k8_v<V>_t<T>
  core="${base#output_}"
  size="${core%%_*}"                 # take the first field as <size>

  in_list "$size" "${SIZES_LIST}" || continue

  ms="$(ms_from_file "$f")"; : "${ms:=0}"
  thr="$(awk -v n="$size" -v t="$ms" 'BEGIN{ if(t>0) printf "%.2f",(n*1000.0)/t; else printf "0"}')"
  echo "$size,$thr" >> "$OUT_CSV"
done

tmp="$(mktemp)"; sort -t, -k1,1n "$OUT_CSV" > "$tmp" && mv "$tmp" "$OUT_CSV"
echo "Wrote $(realpath "$OUT_CSV")"
