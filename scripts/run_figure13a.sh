#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
OUT="$ROOT/out/fig13a"; mkdir -p "$OUT"

HR="$ROOT/pmem-rocksdb/h-rocks"
PM="$ROOT/pmem-rocksdb/benchmarks"
VP="$ROOT/viper/microbenchmarks"
PL="$ROOT/Plush/examples"

[[ -x "$HR/run_ycsbA.sh" ]] && (cd "$HR" && ./run_ycsbA.sh) || echo "[H-Rocks] run_ycsbA.sh not found."
[[ -x "$PM/run_ycsbA.sh" ]] && (cd "$PM" && ./run_ycsbA.sh) || echo "[pmem-rocksdb] run_ycsbA.sh not found."
[[ -x "$VP/run_ycsbA.sh" ]] && (cd "$VP" && ./run_ycsbA.sh) || echo "[Viper] run_ycsbA.sh not found."
[[ -x "$PL/run_ycsbA.sh" ]] && (cd "$PL" && ./run_ycsbA.sh) || echo "[Plush] run_ycsbA.sh not found."

SIZES="${SIZES:-}"
( cd "$HR" && SIZES="$SIZES" ./parse_ycsbA.sh output_ycsbA "$OUT/hrocks_ycsbA.csv" )
( cd "$PM" && SIZES="$SIZES" ./parse_ycsbA.sh output_ycsbA "$OUT/pmem_ycsbA.csv" )
( cd "$VP" && SIZES="$SIZES" ./parse_ycsbA.sh output_ycsbA "$OUT/viper_ycsbA.csv" )
( cd "$PL" && SIZES="$SIZES" ./parse_ycsbA.sh ../output_ycsbA "$OUT/plush_ycsbA.csv" )

python3 "$ROOT/scripts/plot_lines_from_csvs.py" \
  --title "Figure 13a: YCSB-A Throughput vs Size" \
  --xlabel "Operations (N)" \
  --ylabel "Throughput (ops/s)" \
  --out "$OUT/fig13a_ycsbA.pdf" \
  --series "$OUT/hrocks_ycsbA.csv:H-Rocks" \
  --series "$OUT/pmem_ycsbA.csv:pmem-rocksdb" \
  --series "$OUT/viper_ycsbA.csv:Viper" \
  --series "$OUT/plush_ycsbA.csv:Plush"
echo "Done: $OUT"
