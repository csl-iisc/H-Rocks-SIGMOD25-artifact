#!/usr/bin/env bash
# Parse outputs from run_puts_value.sh (value-size sweep) into CSV.
# Usage: ./parse_put_values.sh [INPUT_DIR] [OUTPUT_CSV]
# Defaults: INPUT_DIR=output_put_values  OUTPUT_CSV=put_values_throughput.csv
# Optional: VAL_SIZES="8 16 32" to filter.
set -euo pipefail

IN_DIR="${1:-output_put_values}"
OUT_CSV="${2:-put_values_throughput.csv}"
VAL_FILTER="${VAL_SIZES:-}"

echo "val_size,throughput_ops_per_s" > "$OUT_CSV"
shopt -s nullglob

get_num() { # first numeric (float) after key, or empty
  local key="$1" file="$2" v
  v="$(grep -m1 -E "$key" "$file" 2>/dev/null | grep -Eo '[0-9]+(\.[0-9]+)?' | head -1 || true)"
  [[ -n "${v:-}" ]] && echo "$v" || echo ""
}

for f in "$IN_DIR"/output_*_*_*; do
  [[ -f "$f" ]] || continue
  base="$(basename "$f")"               # output_<n>_<k>_<v>
  n="${base#output_}"; n="${n%%_*}"     # total ops
  kv="${base#output_${n}_}"             # <k>_<v>
  vsize="${kv##*_}"

  if [[ -n "$VAL_FILTER" && " $VAL_FILTER " != *" $vsize "* ]]; then
    continue
  fi

  ops="$(get_num 'Number of puts' "$f")"; [[ -z "$ops" ]] && ops="$n"

  # Collect timings (ms) if present and sum them.
  setup_ms="$(get_num 'setup_time' "$f")"; : "${setup_ms:=0}"
  sort_ms="$(get_num 'sort_time' "$f")"; : "${sort_ms:=0}"
  write_ms="$(get_num 'write_batch_time' "$f")"; : "${write_ms:=0}"
  memtbl_ms="$(get_num 'memtable_time' "$f")"; : "${memtbl_ms:=0}"
  value_copy_ms="$(get_num 'value_copy_time' "$f")"; : "${value_copy_ms:=0}"
  sst_insert_ms="$(get_num 'sst_insertion_time' "$f")"; : "${sst_insert_ms:=0}"
  sst_setup_ms="$(get_num 'sst_setup_time' "$f")"; : "${sst_setup_ms:=0}"
  sst_file_ms="$(get_num 'sstFileTime' "$f")"; : "${sst_file_ms:=0}"

  total_ms="$(awk -v a="$setup_ms" -v b="$sort_ms" -v c="$write_ms" -v d="$memtbl_ms" \
                   -v e="$value_copy_ms" -v f1="$sst_insert_ms" -v g="$sst_setup_ms" -v h="$sst_file_ms" \
                   'BEGIN{printf "%.6f",(a+b+c+d+e+f1+g+h)}')"
  thr_ops="$(awk -v n="$ops" -v ms="$total_ms" 'BEGIN{if(ms>0)printf "%.2f",(n*1000.0)/ms; else printf "0"}')"

  echo "$vsize,$thr_ops" >> "$OUT_CSV"
done

tmp="$(mktemp)"; sort -t, -k1,1n "$OUT_CSV" > "$tmp" && mv "$tmp" "$OUT_CSV"
echo "Wrote $(realpath "$OUT_CSV")"
