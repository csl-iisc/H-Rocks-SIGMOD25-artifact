#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
OUT="$ROOT/out/fig8b"; mkdir -p "$OUT"

HR="$ROOT/pmem-rocksdb/h-rocks"
PM="$ROOT/pmem-rocksdb/examples"
VP="$ROOT/viper/microbenchmarks"
PL="$ROOT/Plush/examples"

[[ -x "$HR/run_gets.sh" ]] && (cd "$HR" && ./run_gets.sh) || echo "[H-Rocks] run_gets.sh not found."
[[ -x "$PM/run_gets.sh" ]] && (cd "$PM" && ./run_gets.sh) || echo "[pmem-rocksdb] run_gets.sh not found."
[[ -x "$VP/run_gets.sh" ]] && (cd "$VP" && ./run_gets.sh) || echo "[Viper] run_gets.sh not found."
[[ -x "$PL/run_gets.sh" ]] && (cd "$PL" && ./run_gets.sh) || echo "[Plush] run_gets.sh not found."

SIZES="${SIZES:-}"; VAL_SIZES="${VAL_SIZES:-8}"
( cd "$HR" && SIZES="$SIZES" ./parse_gets.sh . "$OUT/hrocks_gets.csv" )
( cd "$PM" && SIZES="$SIZES" ./pmem_parse_gets.sh output_gets "$OUT/pmem_gets.csv" )
( cd "$VP" && SIZES="$SIZES" VAL_SIZES="$VAL_SIZES" ./viper_parse_gets.sh output_gets "$OUT/viper_gets.csv" )
( cd "$PL" && SIZES="$SIZES" ./plush_parse_gets.sh . "$OUT/plush_gets.csv" )

python3 "$ROOT/scripts/plot_lines_from_csvs.py" \
  --title "Figure 8b: GETS Throughput vs Size" \
  --xlabel "Operations (N)" \
  --ylabel "Throughput (ops/s)" \
  --val-size "$VAL_SIZES" \
  --out "$OUT/fig8b_gets.pdf" \
  --series "$OUT/hrocks_gets.csv:H-Rocks" \
  --series "$OUT/pmem_gets.csv:pmem-rocksdb" \
  --series "$OUT/viper_gets.csv:Viper" \
  --series "$OUT/plush_gets.csv:Plush"
echo "Done: $OUT"
