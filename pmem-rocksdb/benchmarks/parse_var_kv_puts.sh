#!/usr/bin/env bash
# Usage: ./pmem_parse_var_kv_puts.sh [INPUT_DIR] [OUTPUT_CSV]
# Ex:    ./pmem_parse_var_kv_puts.sh output_var_kv_puts pmem_var_kv_puts.csv
# Filters (optional env): SIZES="10000 1000000" KEYS="8 16" VALS="8 32 128"
set -euo pipefail

IN_DIR="${1:-output_var_kv_puts}"
OUT_CSV="${2:-pmem_var_kv_puts.csv}"
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

for f in "$IN_DIR"/puts_*.log; do
  base="$(basename "$f")"                    # puts_<N>_<K>_<V>.log  (or puts_<N>.log)
  core="${base#puts_}"; core="${core%.log}"

  IFS_= read -r N K V <<< "$core"
  K="${K:-8}"; V="${V:-8}"                  # fallback if only N present

  in_list "$N" "${SIZES_LIST}" || continue
  in_list "$K" "${KEYS_LIST}"  || continue
  in_list "$V" "${VALS_LIST}"  || continue

  ms="$(ms_from_key 'prefill_time' "$f")"; : "${ms:=0}"
  thr="$(awk -v n="$N" -v t="$ms" 'BEGIN{ if(t>0) printf "%.2f",(n*1000.0)/t; else printf "0"}')"
  echo "${N},${K},${V},${thr}" >> "$OUT_CSV"
done

tmp="$(mktemp)"; sort -t, -k1,1n -k2,2n -k3,3n "$OUT_CSV" > "$tmp" && mv "$tmp" "$OUT_CSV"
echo "Wrote $(realpath "$OUT_CSV")"
