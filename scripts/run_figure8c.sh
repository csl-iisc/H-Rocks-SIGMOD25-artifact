#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
OUT="$ROOT/out/fig8c"; mkdir -p "$OUT"

HR="$ROOT/pmem-rocksdb/h-rocks"
PM="$ROOT/pmem-rocksdb/examples"
VP="$ROOT/viper/microbenchmarks"
PL="$ROOT/Plush/examples"

[[ -x "$HR/run_deletes.sh" ]] && (cd "$HR" && ./run_deletes.sh) || echo "[H-Rocks] run_deletes.sh not found."
[[ -x "$PM/run_deletes.sh" ]] && (cd "$PM" && ./run_deletes.sh) || echo "[pmem-rocksdb] run_deletes.sh not found."
[[ -x "$VP/run_deletes.sh" ]] && (cd "$VP" && ./run_deletes.sh) || echo "[Viper] run_deletes.sh not found."
[[ -x "$PL/run_deletes.sh" ]] && (cd "$PL" && ./run_deletes.sh) || echo "[Plush] run_deletes.sh not found."

SIZES="${SIZES:-}"
( cd "$HR" && SIZES="$SIZES" ./parse_deletes_min.sh . "$OUT/hrocks_deletes.csv" )
( cd "$PM" && SIZES="$SIZES" ./pmem_parse_deletes_min.sh output_deletes "$OUT/pmem_deletes.csv" )
( cd "$VP" && SIZES="$SIZES" ./viper_parse_deletes_min.sh output_deletes "$OUT/viper_deletes.csv" )
( cd "$PL" && SIZES="$SIZES" ./plush_parse_deletes_min.sh . "$OUT/plush_deletes.csv" )

python3 "$ROOT/scripts/plot_lines_from_csvs.py" \
  --title "Figure 8c: DELETE Throughput vs Size" \
  --xlabel "Operations (N)" \
  --ylabel "Throughput (ops/s)" \
  --out "$OUT/fig8c_deletes.pdf" \
  --series "$OUT/hrocks_deletes.csv:H-Rocks" \
  --series "$OUT/pmem_deletes.csv:pmem-rocksdb" \
  --series "$OUT/viper_deletes.csv:Viper" \
  --series "$OUT/plush_deletes.csv:Plush"
echo "Done: $OUT"
