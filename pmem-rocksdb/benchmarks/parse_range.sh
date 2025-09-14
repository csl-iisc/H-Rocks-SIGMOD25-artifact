#!/usr/bin/env bash
# Usage:
#   ./pmem_parse_range.sh [INPUT_DIR] [OUTPUT_CSV]
# Example:
#   ./pmem_parse_range.sh output_range pmem_range_throughput.csv
# Optional filters:
#   SIZES="10000 100000 1000000" ./pmem_parse_range.sh

set -euo pipefail
IN_DIR="${1:-output_range}"
OUT_CSV="${2:-pmem_range_throughput.csv}"
SIZES_LIST="${SIZES:-}"

echo "size,throughput_ops_per_s" > "$OUT_CSV"
shopt -s nullglob

in_list(){ [[ -z "${2:-}" ]] && return 0; [[ " $2 " == *" $1 "* ]]; }

# Extract "<number> ms" from the line containing the key
ms_from_key(){ awk -v key="$1" '$0~key{ if (match($0,/([0-9]+(\.[0-9]+)?) ms/,a)) {print a[1]; exit} }' "$2"; }
# Extract "range_rate: <number>" (ranges/sec) if present
rate_from_file(){ sed -nE 's/.*range_rate:\s*([0-9]+(\.[0-9]+)?).*/\1/p;q' "$1"; }

for f in "$IN_DIR"/range_*.log; do
  base="$(basename "$f")"          # range_<N>.log
  size="${base#range_}"; size="${size%.log}"

  in_list "$size" "${SIZES_LIST}" || continue

  rate="$(rate_from_file "$f")"
  if [[ -n "${rate:-}" ]]; then
    thr="$rate"
  else
    ms="$(ms_from_key 'range_time' "$f")"; : "${ms:=0}"
    thr="$(awk -v n="$size" -v t="$ms" 'BEGIN{ if(t>0) printf "%.6f",(n*1000.0)/t; else printf "0"}')"
  fi
  echo "$size,$thr" >> "$OUT_CSV"
done

tmp="$(mktemp)"; sort -t, -k1,1n "$OUT_CSV" > "$tmp" && mv "$tmp" "$OUT_CSV"
echo "Wrote $(realpath "$OUT_CSV")"
