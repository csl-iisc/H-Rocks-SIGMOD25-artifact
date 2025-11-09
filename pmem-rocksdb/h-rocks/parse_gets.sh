#!/usr/bin/env bash
# parse_gets.sh
# Usage:
#   ./parse_gets.sh [INPUT_DIR] [OUTPUT_CSV]
# Example:
#   ./parse_gets.sh               # auto-detects output_get_put, output_gets, then .
#   ./parse_gets.sh output_gets get_throughput.csv
#
# Notes:
# - Looks for legacy GPU logs: output_50M_<ksz>_<vsz>_<size>
# - Also handles new CPU logs: get_put_k<ksz>_v<vsz>_n<size>.log inside output_get_put
# - If $SIZES is set (space-separated), only those sizes are included.

set -euo pipefail

IN_ARG="${1:-}"
OUT_CSV="${2:-get_throughput.csv}"
SIZES_LIST="${SIZES:-}"   # optional whitelist, e.g. "100000 200000 400000"

if [[ -n "$IN_ARG" ]]; then
  IN_DIR="$IN_ARG"
else
  for candidate in output_get_put output_gets .; do
    if [[ -d "$candidate" ]]; then
      IN_DIR="$candidate"
      break
    fi
  done
fi

if [[ -z "${IN_DIR:-}" || ! -d "$IN_DIR" ]]; then
  echo "Input directory '${IN_ARG:-output_get_put}' not found." >&2
  exit 1
fi

echo "size,throughput_ops_per_s" > "$OUT_CSV"
shopt -s nullglob

# Extract last numeric (float) on the matching line; 0 if not found
get_num() {
  local key="$1" file="$2"
  local v
  v="$(grep -m1 -E "$key" "$file" 2>/dev/null | grep -Eo '[0-9]+(\.[0-9]+)?' | tail -1 || true)"
  [[ -n "${v:-}" ]] && echo "$v" || echo "0"
}

should_include_size() {
  local size="$1"
  if [[ -z "$SIZES_LIST" ]]; then
    return 0
  fi
  for s in $SIZES_LIST; do
    if [[ "$size" == "$s" ]]; then
      return 0
    fi
  done
  return 1
}

emit_row() {
  local file="$1" size="$2"
  if ! should_include_size "$size"; then
    return
  fi
  local rs rk rb cb total_ms thr_ops_s
  rs="$(get_num 'read_setup_time'  "$file")"
  rk="$(get_num 'read_kernel_time' "$file")"
  rb="$(get_num 'read_batch_time'  "$file")"
  cb="$(get_num 'copy_back_time'   "$file")"

  total_ms="$(awk -v a="$rs" -v b="$rk" -v c="$rb" -v d="$cb" 'BEGIN{printf "%.6f",(a+b+c+d)}')"
  thr_ops_s="$(awk -v n="$size" -v ms="$total_ms" 'BEGIN{if(ms>0){val=(n*1000.0)/ms;if(val>n)val=n;printf "%.2f",val}else printf "0"}')"

  echo "${size},${thr_ops_s}" >> "$OUT_CSV"
}

# Legacy GPU logs
for f in "$IN_DIR"/output_50M_*_*_*; do
  base="$(basename "$f")"         # e.g., output_50M_8_8_100000
  size="${base##*_}"              # -> 100000
  emit_row "$f" "$size"
done

# New CPU logs from run_gets.sh (get_put_k<k>_v<v>_n<size>.log)
for f in "$IN_DIR"/get_put_k*_v*_n*.log; do
  base="$(basename "$f")"
  if [[ "$base" =~ get_put_k([0-9]+)_v([0-9]+)_n([0-9]+)\.log$ ]]; then
    size="${BASH_REMATCH[3]}"
    emit_row "$f" "$size"
  fi
done

# sort numerically by size
tmp="$(mktemp)"
sort -t, -k1,1n "$OUT_CSV" > "$tmp" && mv "$tmp" "$OUT_CSV"

echo "Wrote $(realpath "$OUT_CSV")"
