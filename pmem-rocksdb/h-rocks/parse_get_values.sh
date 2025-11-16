#!/usr/bin/env bash
# Parse GET value-size sweeps into CSV.
# Usage: ./parse_get_values.sh [INPUT_DIR] [OUTPUT_CSV]
# Defaults: INPUT_DIR=output_get_values  OUTPUT_CSV=get_values_throughput.csv
# Optional filter: VAL_SIZES="8 16"
set -euo pipefail

IN_DIR="${1:-output_get_values}"
OUT_CSV="${2:-get_values_throughput.csv}"
VAL_FILTER="${VAL_SIZES:-}"

echo "val_size,throughput_ops_per_s" > "$OUT_CSV"
shopt -s nullglob

get_num() { # last numeric on a matching line from a file (or 0)
  local key="$1" file="$2"
  local v
  v="$(grep -m1 -E "$key" "$file" 2>/dev/null | grep -Eo '[0-9]+(\.[0-9]+)?' | tail -1 || true)"
  [[ -n "${v:-}" ]] && echo "$v" || echo "0"
}

should_include_val() {
  local v="$1"
  if [[ -z "$VAL_FILTER" ]]; then return 0; fi
  for s in $VAL_FILTER; do [[ "$v" == "$s" ]] && return 0; done
  return 1
}

for f in "$IN_DIR"/get_put_k*_v*_n*.log; do
  base="$(basename "$f")"
  if [[ "$base" =~ get_put_k([0-9]+)_v([0-9]+)_n([0-9]+)\.log$ ]]; then
    vsize="${BASH_REMATCH[2]}"
    n="${BASH_REMATCH[3]}"
  else
    continue
  fi

  should_include_val "$vsize" || continue

  rs="$(get_num 'read_setup_time'  "$f")"
  rk="$(get_num 'read_kernel_time' "$f")"
  rb="$(get_num 'read_batch_time'  "$f")"
  cb="$(get_num 'copy_back_time'   "$f")"

  total_ms="$(awk -v a="$rs" -v b="$rk" -v c="$rb" -v d="$cb" 'BEGIN{printf "%.6f",(a+b+c+d)}')"
  thr_ops_s="$(awk -v n="$n" -v ms="$total_ms" 'BEGIN{if(ms>0){val=(n*1000.0)/ms;if(val>n)val=n;printf "%.2f",val}else printf "0"}')"

  echo "${vsize},${thr_ops_s}" >> "$OUT_CSV"
done

tmp="$(mktemp)"; sort -t, -k1,1n "$OUT_CSV" > "$tmp" && mv "$tmp" "$OUT_CSV"
echo "Wrote $(realpath "$OUT_CSV")"
