#!/usr/bin/env bash
# Usage: ./parse_atomic_txn.sh [INPUT_DIR] [OUTPUT_CSV]
#   INPUT_DIR defaults to output_atomic_txn (written by run_atomic_txn.sh)
#   OUTPUT_CSV defaults to atomic_txn_throughput.csv
# Optional filter: SIZES="10000 100000 1000000" ./parse_atomic_txn.sh
set -euo pipefail

IN_DIR="${1:-output_atomic_txn}"
OUT_CSV="${2:-atomic_txn_throughput.csv}"
SIZES_LIST="${SIZES:-}"

echo "size,throughput_ops_per_s" > "$OUT_CSV"
shopt -s nullglob

get_elapsed() {
  local f="$1"
  grep -m1 -E 'elapsed_s' "$f" 2>/dev/null | awk '{print $NF}' || echo ""
}

for f in "$IN_DIR"/output_*_*; do
  [[ -e "$f" ]] || continue
  base="$(basename "$f")"            # output_<prefill>_<size>
  core="${base#output_}"
  prefill="${core%%_*}"
  size="${core##*_}"

  # Optional size filter
  if [[ -n "$SIZES_LIST" ]]; then
    ok=false
    for s in $SIZES_LIST; do [[ "$size" == "$s" ]] && ok=true; done
    $ok || continue
  fi

  elapsed="$(get_elapsed "$f")"
  [[ -n "${elapsed:-}" ]] || { echo "warn: elapsed time missing in $f; skipping" >&2; continue; }

  thr_ops="$(awk -v n="$prefill" -v e="$elapsed" 'BEGIN{ if(e>0) printf "%.2f", n/e; else print "0" }')"
  echo "$size,$thr_ops" >> "$OUT_CSV"
done

tmp="$(mktemp)"; sort -t, -k1,1n "$OUT_CSV" > "$tmp" && mv "$tmp" "$OUT_CSV"
echo "Wrote $(realpath "$OUT_CSV")"
