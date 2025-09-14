#!/usr/bin/env bash
# Usage: ./pmem_parse_var_kv_gets.sh [INPUT_DIR] [OUTPUT_CSV]
# Ex:    ./pmem_parse_var_kv_gets.sh output_var_kv_gets pmem_var_kv_gets.csv
# Filters (optional env): SIZES="10000 1000000" KEYS="8 16" VALS="8 32 128"
set -euo pipefail

IN_DIR="${1:-output_var_kv_gets}"
OUT_CSV="${2:-pmem_var_kv_gets.csv}"
SIZES_LIST="${SIZES:-}"
KEYS_LIST="${KEYS:-}"
VALS_LIST="${VALS:-}"

echo "size,k,v,throughput_ops_per_s" > "$OUT_CSV"
shopt -s nullglob

# extract "<number> ms" from the line containing the key
ms_from_key(){ awk -v key="$1" '$0~key{ if (match($0,/([0-9]+(\.[0-9]+)?) ms/,a)) {print a[1]; exit} }' "$2"; }

in_list(){ # $1=val $2=list
  [[ -z "${2:-}" ]] && return 0
  [[ " $2 " == *" $1 "* ]]
}

for f in "$IN_DIR"/gets_*.log; do
  base="$(basename "$f")"                    # gets_<N>_<K>_<V>.log  (or gets_<N>.log)
  core="${base#gets_}"; core="${core%.log}"

  IFS=_ read -r N K V <<< "$core"
  K="${K:-8}"; V="${V:-8}"                  # fallback if only N present

  in_list "$N" "${SIZES_LIST}" || continue
  in_list "$K" "${KEYS_LIST}"  || continue
  in_list "$V" "${VALS_LIST}"  || continue

  ms="$(ms_from_key 'get_time' "$f")"; : "${ms:=0}"
  thr="$(awk -v n="$N" -v t="$ms" 'BEGIN{ if(t>0) printf "%.2f",(n*1000.0)/t; else printf "0"}')"
  echo "${N},${K},${V},${thr}" >> "$OUT_CSV"
done

tmp="$(mktemp)"; sort -t, -k1,1n -k2,2n -k3,3n "$OUT_CSV" > "$tmp" && mv "$tmp" "$OUT_CSV"
echo "Wrote $(realpath "$OUT_CSV")"
