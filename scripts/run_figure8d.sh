#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
OUT="$ROOT/out/fig8d"; mkdir -p "$OUT"

HR="$ROOT/pmem-rocksdb/h-rocks"
PM="$ROOT/pmem-rocksdb/benchmarks"
VP="$ROOT/viper/microbenchmarks"
PL="$ROOT/Plush/examples"
UT="$ROOT/utree/multiThread/utree"

[[ -x "$HR/run_updates.sh" ]] && (cd "$HR" && ./run_updates.sh) || echo "[H-Rocks] run_updates.sh not found."
[[ -x "$PM/run_updates.sh" ]] && (cd "$PM" && ./run_updates.sh) || echo "[pmem-rocksdb] run_updates.sh not found."
[[ -x "$VP/run_update.sh" ]] && (cd "$VP" && ./run_update.sh) || echo "[Viper] run_update.sh not found."
[[ -x "$PL/run_update.sh" ]] && (cd "$PL" && ./run_update.sh) || echo "[Plush] run_update.sh not found."
[[ -x "$UT/run_updates.sh" ]] && (cd "$UT" && ./run_updates.sh) || echo "[uTree] run_updates.sh not found."

SIZES="${SIZES:-}"
( cd "$HR" && SIZES="$SIZES" ./parse_updates.sh . "$OUT/hrocks_updates.csv" )
( cd "$PM" && SIZES="$SIZES" ./parse_updates.sh output_updates "$OUT/pmem_updates.csv" )
( cd "$VP" && SIZES="$SIZES" ./parse_update.sh output_updates "$OUT/viper_updates.csv" )
( cd "$PL" && SIZES="$SIZES" ./parse_updates.sh ../output_updates "$OUT/plush_updates.csv" )
( cd "$UT" && SIZES="$SIZES" ./parse_updates.sh output_updates "$OUT/utree_updates.csv" )

python3 "$ROOT/scripts/plot_lines_from_csvs.py" \
  --title "Figure 8d: UPDATE Throughput vs Size" \
  --xlabel "Operations (N)" \
  --ylabel "Throughput (ops/s)" \
  --out "$OUT/fig8d_updates.pdf" \
  --series "$OUT/hrocks_updates.csv:H-Rocks" \
  --series "$OUT/pmem_updates.csv:pmem-rocksdb" \
  --series "$OUT/viper_updates.csv:Viper" \
  --series "$OUT/plush_updates.csv:Plush" \
  --series "$OUT/utree_updates.csv:uTree"
echo "Done: $OUT"
