#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
OUT="$ROOT/out/fig8a"; mkdir -p "$OUT"

HR="$ROOT/pmem-rocksdb/h-rocks"
PM="$ROOT/pmem-rocksdb/benchmarks"
VP="$ROOT/viper/microbenchmarks"
PL="$ROOT/Plush/examples"
UT="$ROOT/utree/multiThread/utree"

# 1) (Optional) run experiments if runners are available
[[ -x "$HR/run_puts.sh" ]] && (cd "$HR" && ./run_puts.sh) || echo "[H-Rocks] run_puts.sh not found, skipping run."
[[ -x "$PM/run_puts.sh" ]] && (cd "$PM" && ./run_puts.sh) || echo "[pmem-rocksdb] run_puts.sh not found, skipping run."
[[ -x "$VP/run_prefill_put.sh" ]] && (cd "$VP" && ./run_prefill_put.sh) || echo "[Viper] run_prefill_put.sh not found, skipping run."
[[ -x "$PL/run_put.sh" ]] && (cd "$PL" && ./run_put.sh) || echo "[Plush] run_put.sh not found, skipping run."
[[ -x "$UT/run_inserts.sh" ]] && (cd "$UT" && ./run_inserts.sh) || echo "[uTree] run_inserts.sh not found, skipping run."

# 2) Parse -> CSV (size, throughput_ops_per_s)
SIZES="${SIZES:-}" # forward to parsers if set
( cd "$HR" && SIZES="$SIZES" ./parse_puts.sh output_puts "$OUT/hrocks_puts.csv" )
( cd "$PM" && SIZES="$SIZES" ./parse_puts.sh output_puts "$OUT/pmem_puts.csv" )
( cd "$VP" && SIZES="$SIZES" ./parse_puts.sh output_puts "$OUT/viper_puts.csv" )
( cd "$PL" && SIZES="$SIZES" ./parse_puts.sh . "$OUT/plush_puts.csv" )
( cd "$UT" && SIZES="$SIZES" ./parse_inserts.sh output_inserts "$OUT/utree_puts.csv" )

# 3) Plot
python3 "$ROOT/scripts/plot_lines_from_csvs.py" \
  --title "Figure 8a: PUTS Throughput vs Size" \
  --xlabel "Operations (N)" \
  --ylabel "Throughput (ops/s)" \
  --out "$OUT/fig8a_puts.pdf" \
  --series "$OUT/hrocks_puts.csv:H-Rocks" \
  --series "$OUT/pmem_puts.csv:pmem-rocksdb" \
  --series "$OUT/viper_puts.csv:Viper" \
  --series "$OUT/plush_puts.csv:Plush" \
  --series "$OUT/utree_puts.csv:uTree"
echo "Done: $OUT"
