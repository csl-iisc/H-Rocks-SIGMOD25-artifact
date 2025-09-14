#!/usr/bin/env bash
# Usage: ./plush_parse_var_kv_gets.sh [INPUT_DIR] [OUTPUT_CSV]
# Ex:    ./plush_parse_var_kv_gets.sh output_var_kv_gets plush_var_kv_gets.csv
# Notes: expects files named output_<count>_<k>_<v>
# Env filters (optional): SIZES="5000000" KEYS="8 16" VALS="8 32 128"
set -euo pipefail

IN_DIR="${1:-output_var_kv_gets}"
OUT_CSV="${2:-plush_var_kv_gets.csv}"
SIZES_LIST="${SIZES:-}"
KEYS_LIST="${KEYS:-}"
VALS_LIST="${VALS:-}"

echo "size,k,v,throughput_ops_per_s" > "$OUT_CSV"
shopt -s nullglob

in_list(){ # $1=val $2=list
  [[ -z "${2:-}" ]] && return 0
  [[ " $2 " == *" $1 "* ]]
}

for f in "$IN_DIR"/output_*_*_*; do
  base="$(basename "$f")"                  # output_<N>_<K>_<V>
  IFS_= read -r _ N K V <<< "$base"

  in_list "$N" "${SIZES_LIST}" || continue
  in_list "$K" "${KEYS_LIST}"  || continue
  in_list "$V" "${VALS_LIST}"  || continue

  # grab last field on the first 'get_time' line (assumed milliseconds)
  ms="$(grep -m1 -E 'get_time' "$f" 2>/dev/null | awk '{print $NF}')"; : "${ms:=0}"

  thr="$(awk -v n="$N" -v t="$ms" 'BEGIN{ if(t>0) printf "%.2f",(n*1000.0)/t; else printf "0"}')"
  echo "${N},${K},${V},${thr}" >> "$OUT_CSV"
done

tmp="$(mktemp)"; sort -t, -k1,1n -k2,2n -k3,3n "$OUT_CSV" > "$tmp" && mv "$tmp" "$OUT_CSV"
echo "Wrote $(realpath "$OUT_CSV")"
