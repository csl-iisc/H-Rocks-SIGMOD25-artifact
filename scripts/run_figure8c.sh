#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
OUT="$ROOT/out/fig8c"; mkdir -p "$OUT"

HR="$ROOT/pmem-rocksdb/h-rocks"
PM="$ROOT/pmem-rocksdb/benchmarks"
VP="$ROOT/viper/microbenchmarks"
PL="$ROOT/Plush/examples"
UT="$ROOT/utree/multiThread/utree"

[[ -x "$HR/run_deletes.sh" ]] && (cd "$HR" && ./run_deletes.sh) || echo "[H-Rocks] run_deletes.sh not found."
[[ -x "$PM/run_deletes.sh" ]] && (cd "$PM" && ./run_deletes.sh) || echo "[pmem-rocksdb] run_deletes.sh not found."
[[ -x "$VP/run_prefill_delete.sh" ]] && (cd "$VP" && ./run_prefill_delete.sh) || echo "[Viper] run_prefill_delete.sh not found."
[[ -x "$PL/run_delete.sh" ]] && (cd "$PL" && ./run_delete.sh) || echo "[Plush] run_delete.sh not found."
[[ -x "$UT/run_removes.sh" ]] && (cd "$UT" && ./run_removes.sh) || echo "[uTree] run_removes.sh not found."

SIZES="${SIZES:-}"
( cd "$HR" && SIZES="$SIZES" ./parse_deletes.sh . "$OUT/hrocks_deletes.csv" )
( cd "$PM" && SIZES="$SIZES" ./parse_deletes.sh output_deletes "$OUT/pmem_deletes.csv" )
( cd "$VP" && SIZES="$SIZES" ./parse_delete.sh output_deletes "$OUT/viper_deletes.csv" )
( cd "$PL" && SIZES="$SIZES" ./parse_deletes.sh . "$OUT/plush_deletes.csv" )
( cd "$UT" && SIZES="$SIZES" ./parse_removes.sh output_removes "$OUT/utree_deletes.csv" )

python3 "$ROOT/scripts/plot_lines_from_csvs.py" \
  --title "Figure 8c: DELETE Throughput vs Size" \
  --xlabel "Operations (N)" \
  --ylabel "Throughput (ops/s)" \
  --out "$OUT/fig8c_deletes.pdf" \
  --series "$OUT/hrocks_deletes.csv:H-Rocks" \
  --series "$OUT/pmem_deletes.csv:pmem-rocksdb" \
  --series "$OUT/viper_deletes.csv:Viper" \
  --series "$OUT/plush_deletes.csv:Plush" \
  --series "$OUT/utree_deletes.csv:uTree"
echo "Done: $OUT"
