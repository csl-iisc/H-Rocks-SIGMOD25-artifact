#!/usr/bin/env bash
# plush_parse_ycsbC.sh
set -euo pipefail
IN_DIR="${1:-.}"
OUT_CSV="${2:-plush_ycsbC_throughput.csv}"
SIZES_LIST="${SIZES:-}"

echo "size,throughput_ops_per_s" > "$OUT_CSV"
shopt -s nullglob

extract_time_ms() {
  local line="$1" value unit_ms
  [[ -z "$line" ]] && { echo "0"; return; }
  if [[ "$line" =~ ([0-9]+(\.[0-9]+)?) ]]; then
    value="${BASH_REMATCH[1]}"
  else
    echo "0"; return
  fi
  if [[ "$line" == *" s"* ]]; then
    unit_ms="$(awk -v v="$value" 'BEGIN{printf "%.6f", v*1000.0}')"
    echo "$unit_ms"
  else
    echo "$value"
  fi
}

for f in "$IN_DIR"/output_*; do
  [[ -f "$f" ]] || continue
  base="$(basename "$f")"
  if [[ "$base" =~ ^output_([0-9]+)_ ]]; then
    size="${BASH_REMATCH[1]}"
  else
    continue
  fi
  [[ -n "$SIZES_LIST" && " $SIZES_LIST " != *" $size "* ]] && continue

  line="$(grep -m1 -E 'read_time' "$f" 2>/dev/null || true)"
  ms="$(extract_time_ms "$line")"
  thr="$(awk -v n="$size" -v t="$ms" 'BEGIN{ if(t>0){val=(n*1000.0)/t; if(val>n) val=n; printf "%.2f",val} else printf "0"}')"
  echo "$size,$thr" >> "$OUT_CSV"
done

tmp="$(mktemp)"
sort -t, -k1,1n "$OUT_CSV" > "$tmp" && mv "$tmp" "$OUT_CSV"
echo "Wrote $(realpath "$OUT_CSV")"
