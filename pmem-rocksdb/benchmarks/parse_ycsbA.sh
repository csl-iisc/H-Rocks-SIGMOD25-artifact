#!/usr/bin/env bash
# pmem_parse_ycsbA.sh
# Usage: ./pmem_parse_ycsbA.sh [INPUT_DIR] [OUTPUT_CSV]
# Ex:    ./pmem_parse_ycsbA.sh output_ycsbA pmem_ycsbA_throughput.csv
set -euo pipefail
IN_DIR="${1:-output_ycsbA}"
OUT_CSV="${2:-pmem_ycsbA_throughput.csv}"
SIZES_LIST="${SIZES:-}"

echo "size,throughput_ops_per_s" > "$OUT_CSV"; shopt -s nullglob

calc_thr_from_log() {
  local logfile="$1" size="$2"
  local run_line mops run_ms thr
  run_line="$(grep -m1 -E '^run_time:' "$logfile" || true)"
  if [[ -n "$run_line" && "$run_line" =~ throughput:[[:space:]]*([0-9.]+) ]]; then
    mops="${BASH_REMATCH[1]}"
    if [[ -n "$mops" ]]; then
      thr="$(awk -v m="$mops" 'BEGIN{printf "%.2f", m*1000000.0}')"
    fi
  fi
  if [[ -z "${thr:-}" || "$thr" == "0" || "$thr" == "0.00" ]]; then
    if [[ -n "$run_line" && "$run_line" =~ run_time:[[:space:]]*([0-9.]+) ]]; then
      run_ms="${BASH_REMATCH[1]}"
    else
      run_ms="$(grep -m1 -Eo 'run_time:[[:space:]]*[0-9]+(\.[0-9]+)?' "$logfile" 2>/dev/null | awk '{print $2}')"
    fi
    if [[ -n "$run_ms" && "$run_ms" != "0" ]]; then
      thr="$(awk -v n="$size" -v t="$run_ms" 'BEGIN{if(t>0){val=(n*1000.0)/t;if(val>n)val=n;printf "%.2f",val}else printf "0"}')"
    fi
  fi
  echo "${thr:-0}"
}

for f in "$IN_DIR"/ycsbA_*.log; do
  base="$(basename "$f")"; size="${base#ycsbA_}"; size="${size%.log}"
  if [[ -n "$SIZES_LIST" && " $SIZES_LIST " != *" $size "* ]]; then continue; fi
  thr="$(calc_thr_from_log "$f" "$size")"
  echo "$size,$thr" >> "$OUT_CSV"
done

tmp="$(mktemp)"; sort -t, -k1,1n "$OUT_CSV" > "$tmp" && mv "$tmp" "$OUT_CSV"; echo "Wrote $(realpath "$OUT_CSV")"
