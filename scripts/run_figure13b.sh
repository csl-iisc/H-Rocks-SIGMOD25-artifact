#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
OUT="$ROOT/out/fig13b"; mkdir -p "$OUT"

HR="$ROOT/pmem-rocksdb/h-rocks"
PM="$ROOT/pmem-rocksdb/benchmarks"
VP="$ROOT/viper/microbenchmarks"
PL="$ROOT/Plush/examples"

[[ -x "$HR/run_ycsbB.sh" ]] && (cd "$HR" && ./run_ycsbB.sh) || echo "[H-Rocks] run_ycsbB.sh not found."
[[ -x "$PM/run_ycsbB.sh" ]] && (cd "$PM" && ./run_ycsbB.sh) || echo "[pmem-rocksdb] run_ycsbB.sh not found."
[[ -x "$VP/run_ycsbB.sh" ]] && (cd "$VP" && ./run_ycsbB.sh) || echo "[Viper] run_ycsbB.sh not found."
[[ -x "$PL/run_ycsbB.sh" ]] && (cd "$PL" && ./run_ycsbB.sh) || echo "[Plush] run_ycsbB.sh not found."

SIZES="${SIZES:-}"
( cd "$HR" && SIZES="$SIZES" ./parse_ycsbB.sh output_ycsbB "$OUT/hrocks_ycsbB.csv" )
( cd "$PM" && SIZES="$SIZES" ./parse_ycsbB.sh output_ycsbB "$OUT/pmem_ycsbB.csv" )
( cd "$VP" && SIZES="$SIZES" ./parse_ycsbB.sh output_ycsbB "$OUT/viper_ycsbB.csv" )
( cd "$PL" && SIZES="$SIZES" ./parse_ycsbB.sh ../output_ycsbB "$OUT/plush_ycsbB.csv" )

python3 "$ROOT/scripts/plot_lines_from_csvs.py" \
  --title "Figure 13b: YCSB-B Throughput vs Size" \
  --xlabel "Operations (N)" \
  --ylabel "Throughput (ops/s)" \
  --out "$OUT/fig13b_ycsbB.pdf" \
  --series "$OUT/hrocks_ycsbB.csv:H-Rocks" \
  --series "$OUT/pmem_ycsbB.csv:pmem-rocksdb" \
  --series "$OUT/viper_ycsbB.csv:Viper" \
  --series "$OUT/plush_ycsbB.csv:Plush"
echo "Done: $OUT"
