#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
OUT="$ROOT/out/fig13d"; mkdir -p "$OUT"

HR="$ROOT/pmem-rocksdb/h-rocks"
PM="$ROOT/pmem-rocksdb/benchmarks"
VP="$ROOT/viper/microbenchmarks"
PL="$ROOT/Plush/examples"

[[ -x "$HR/run_ycsbD.sh" ]] && (cd "$HR" && ./run_ycsbD.sh) || echo "[H-Rocks] run_ycsbD.sh not found."
[[ -x "$PM/run_ycsbD.sh" ]] && (cd "$PM" && ./run_ycsbD.sh) || echo "[pmem-rocksdb] run_ycsbD.sh not found."
[[ -x "$VP/run_ycsbD.sh" ]] && (cd "$VP" && ./run_ycsbD.sh) || echo "[Viper] run_ycsbD.sh not found."
[[ -x "$PL/run_ycsbD.sh" ]] && (cd "$PL" && ./run_ycsbD.sh) || echo "[Plush] run_ycsbD.sh not found."

SIZES="${SIZES:-}"
( cd "$HR" && SIZES="$SIZES" ./parse_ycsbD.sh output_ycsbD "$OUT/hrocks_ycsbD.csv" )
( cd "$PM" && SIZES="$SIZES" ./parse_ycsbD.sh output_ycsbD "$OUT/pmem_ycsbD.csv" )
( cd "$VP" && SIZES="$SIZES" ./parse_ycsbD.sh output_ycsbD "$OUT/viper_ycsbD.csv" )
( cd "$PL" && SIZES="$SIZES" ./parse_ycsbD.sh ../output_ycsbD "$OUT/plush_ycsbD.csv" )

python3 "$ROOT/scripts/plot_lines_from_csvs.py" \
  --title "Figure 13d: YCSB-D Throughput vs Size" \
  --xlabel "Operations (N)" \
  --ylabel "Throughput (ops/s)" \
  --out "$OUT/fig13d_ycsbD.pdf" \
  --series "$OUT/hrocks_ycsbD.csv:H-Rocks" \
  --series "$OUT/pmem_ycsbD.csv:pmem-rocksdb" \
  --series "$OUT/viper_ycsbD.csv:Viper" \
  --series "$OUT/plush_ycsbD.csv:Plush"
echo "Done: $OUT"
