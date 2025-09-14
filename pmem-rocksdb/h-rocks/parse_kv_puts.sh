#!/usr/bin/env bash
# Parse PUT results for specific KV pairs and compute MOps/s
# Usage: ./parse_puts_kv_min.sh [IN_DIR] [OUT_CSV]
# Default IN_DIR=output_diff_sizes_values  OUT_CSV=rocksdb_puts_kv.csv
set -euo pipefail

IN_DIR="${1:-output_diff_sizes_values}"
OUT_CSV="${2:-rocksdb_puts_kv.csv}"

# only these K/V pairs
WANT=("8/8" "16/32" "16/128" "32/256" "64/128" "128/1024")

echo "kv,count,throughput_mops" > "$OUT_CSV"
shopt -s nullglob

in_want() {
  local kv="$1"
  for w in "${WANT[@]}"; do [[ "$kv" == "$w" ]] && return 0; done
  return 1
}

# helper: extract last numeric on the first matching line; default 0
get_num() {
  local key="$1" file="$2" v
  v="$(grep -m1 -E "$key" "$file" 2>/dev/null | grep -Eo '[0-9]+(\.[0-9]+)?' | tail -1 || true)"
  [[ -n "${v:-}" ]] && echo "$v" || echo "0"
}

for f in "$IN_DIR"/output_*_*_*; do
  base="$(basename "$f")"             # output_<count>_<k>_<v>
  IFS=_ read -r _ count k v <<< "$base"
  kv="${k}/${v}"
  in_want "$kv" || continue

  setup_ms="$(get_num 'setup_time' "$f")"
  sort_ms="$(get_num 'sort_time' "$f")"
  write_ms="$(get_num 'write_batch_time' "$f")"
  memtbl_ms="$(get_num 'memtable_time' "$f")"
  total_ms="$(awk -v a="$setup_ms" -v b="$sort_ms" -v c="$write_ms" -v d="$memtbl_ms" 'BEGIN{printf "%.6f",(a+b+c+d)}')"

  mops="$(awk -v n="$count" -v ms="$total_ms" 'BEGIN{if(ms>0)printf "%.4f",(n/1e6)/(ms/1000.0); else printf "0"}')"
  echo "${kv},${count},${mops}" >> "$OUT_CSV"
done

# sort by kv then count
tmp="$(mktemp)"; sort -t, -k1,1 -k2,2n "$OUT_CSV" > "$tmp" && mv "$tmp" "$OUT_CSV"
echo "Wrote $(realpath "$OUT_CSV")"
