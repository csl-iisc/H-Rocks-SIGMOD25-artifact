#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
OUT="$ROOT/out/fig8a"; mkdir -p "$OUT"

HR="$ROOT/pmem-rocksdb/h-rocks"
PM="$ROOT/pmem-rocksdb/examples"
VP="$ROOT/viper/microbenchmarks"
PL="$ROOT/Plush/examples"

# 1) (Optional) run experiments if you have runners
[[ -x "$HR/run_puts.sh" ]] && (cd "$HR" && ./run_puts.sh) || echo "[H-Rocks] run_puts.sh not found, skipping run."
[[ -x "$PM/run_puts.sh" ]] && (cd "$PM" && ./run_puts.sh) || echo "[pmem-rocksdb] run_puts.sh not found, skipping run."
[[ -x "$VP/run_puts.sh" ]] && (cd "$VP" && ./run_puts.sh) || echo "[Viper] run_puts.sh not found, skipping run."
[[ -x "$PL/run_puts.sh" ]] && (cd "$PL" && ./run_puts.sh) || echo "[Plush] run_puts.sh not found, skipping run."

# 2) Parse -> CSV (size,throughput_ops_per_s)
SIZES="${SIZES:-}" # forward to parsers if set
( cd "$HR" && SIZES="$SIZES" ./parse_puts_min.sh output_puts "$OUT/hrocks_puts.csv" )
( cd "$PM" && SIZES="$SIZES" ./pmem_parse_puts_min.sh output_puts "$OUT/pmem_puts.csv" )
( cd "$VP" && SIZES="$SIZES" ./viper_parse_puts_min.sh output_puts "$OUT/viper_puts.csv" )
( cd "$PL" && SIZES="$SIZES" ./plush_parse_puts_min.sh . "$OUT/plush_puts.csv" )

# 3) Plot
python3 "$ROOT/scripts/plot_lines_from_csvs.py" \
  --title "Figure 8a: PUTS Throughput vs Size" \
  --xlabel "Operations (N)" \
  --ylabel "Throughput (ops/s)" \
  --out "$OUT/fig8a_puts.pdf" \
  --series "$OUT/hrocks_puts.csv:H-Rocks" \
  --series "$OUT/pmem_puts.csv:pmem-rocksdb" \
  --series "$OUT/viper_puts.csv:Viper" \
  --series "$OUT/plush_puts.csv:Plush"
echo "Done: $OUT"
