#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
OUT="$ROOT/out/fig13c"; mkdir -p "$OUT"

HR="$ROOT/pmem-rocksdb/h-rocks"
PM="$ROOT/pmem-rocksdb/examples"
VP="$ROOT/viper/microbenchmarks"
PL="$ROOT/Plush/examples"

[[ -x "$HR/run_ycsbC.sh" ]] && (cd "$HR" && ./run_ycsbC.sh) || echo "[H-Rocks] run_ycsbC.sh not found."
[[ -x "$PM/run_ycsbC.sh" ]] && (cd "$PM" && ./run_ycsbC.sh) || echo "[pmem-rocksdb] run_ycsbC.sh not found."
[[ -x "$VP/run_ycsbC.sh" ]] && (cd "$VP" && ./run_ycsbC.sh) || echo "[Viper] run_ycsbC.sh not found."
[[ -x "$PL/run_ycsbC.sh" ]] && (cd "$PL" && ./run_ycsbC.sh) || echo "[Plush] run_ycsbC.sh not found."

SIZES="${SIZES:-}"
( cd "$HR" && SIZES="$SIZES" ./parse_ycsbC_min.sh . "$OUT/hrocks_ycsbC.csv" )
( cd "$PM" && SIZES="$SIZES" ./pmem_parse_ycsbC_min.sh output_ycsbC "$OUT/pmem_ycsbC.csv" )
( cd "$VP" && SIZES="$SIZES" ./viper_parse_ycsbC_min.sh output_ycsbC "$OUT/viper_ycsbC.csv" )
( cd "$PL" && SIZES="$SIZES" ./plush_parse_ycsbC_min.sh . "$OUT/plush_ycsbC.csv" )

python3 "$ROOT/scripts/plot_lines_from_csvs.py" \
  --title "Figure 13c: YCSB-C Throughput vs Size" \
  --xlabel "Operations (N)" \
  --ylabel "Throughput (ops/s)" \
  --out "$OUT/fig13c_ycsbC.pdf" \
  --series "$OUT/hrocks_ycsbC.csv:H-Rocks" \
  --series "$OUT/pmem_ycsbC.csv:pmem-rocksdb" \
  --series "$OUT/viper_ycsbC.csv:Viper" \
  --series "$OUT/plush_ycsbC.csv:Plush"
echo "Done: $OUT"
