#!/usr/bin/env bash
# Usage:
#   ./viper_parse_var_kv_gets_min.sh [INPUT_DIR] [OUTPUT_CSV]
# Example:
#   ./viper_parse_var_kv_gets_min.sh output_var_kv_gets viper_var_kv_gets.csv
# Optional filters (env):
#   SIZES="1000000 5000000" KEYS="8 16" VALS="8 32 128" ./viper_parse_var_kv_gets_min.sh
set -euo pipefail

IN_DIR="${1:-output_var_kv_gets}"
OUT_CSV="${2:-viper_var_kv_gets.csv}"
SIZES_LIST="${SIZES:-}"
KEYS_LIST="${KEYS:-}"
VALS_LIST="${VALS:-}"

echo "size,k,v,throughput_ops_per_s" > "$OUT_CSV"
shopt -s nullglob

in_list(){ [[ -z "${2:-}" ]] && return 0; [[ " $2 " == *" $1 "* ]]; }

# Try final_get_time first (common in Viper), else get_time fallback
ms_from_get() {
  local f="$1" ms
  ms="$(grep -m1 -E 'final_get_time' "$f" 2>/dev/null | awk '{print $NF}')"
  [[ -n "${ms:-}" ]] || ms="$(grep -m1 -E 'get_time' "$f" 2>/dev/null | awk '{print $NF}')"
  echo "${ms:-0}"
}

for f in "$IN_DIR"/output_*_*_*; do
  base="$(basename "$f")"           # output_<N>_<K>_<V>
  IFS=_ read -r _ N K V <<< "$base"

  in_list "$N" "${SIZES_LIST}" || continue
  in_list "$K" "${KEYS_LIST}"  || continue
  in_list "$V" "${VALS_LIST}"  || continue

  ms="$(ms_from_get "$f")"
  thr="$(awk -v n="$N" -v t="$ms" 'BEGIN{ if(t>0) printf "%.2f",(n*1000.0)/t; else printf "0"}')"
  echo "${N},${K},${V},${thr}" >> "$OUT_CSV"
done

tmp="$(mktemp)"; sort -t, -k1,1n -k2,2n -k3,3n "$OUT_CSV" > "$tmp" && mv "$tmp" "$OUT_CSV"
echo "Wrote $(realpath "$OUT_CSV")"
