#!/usr/bin/env bash
set -euo pipefail

# root + outputs
ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
OUT="$ROOT/out/fig9a"; mkdir -p "$OUT"

# projects
HR="$ROOT/pmem-rocksdb/h-rocks"
PM="$ROOT/pmem-rocksdb/examples"
PL="$ROOT/Plush/examples"

echo "[Fig9a] output dir: $OUT"

############################
# 1) (optional) run sweeps #
############################
if [[ -x "$HR/run_range.sh" ]]; then (cd "$HR" && ./run_range.sh)
else echo "[H-Rocks] run_range.sh not found, skipping run."; fi

if [[ -x "$PM/run_range.sh" ]]; then (cd "$PM" && ./run_range.sh)
else echo "[pmem-rocksdb] run_range.sh not found, skipping run."; fi

if [[ -x "$PL/run_range.sh" ]]; then (cd "$PL" && ./run_range.sh)
else echo "[Plush] run_range.sh not found, skipping run."; fi

#########################################
# 2) Parse logs -> CSVs for each system #
#########################################
SIZES="${SIZES:-}"   # forward to parsers if user sets a filter list

# H-Rocks: accept either parse_range.sh or hrocks_parse_range.sh
if [[ -x "$HR/parse_range.sh" ]]; then
  ( cd "$HR" && SIZES="$SIZES" ./parse_range.sh output_range "$OUT/hrocks_range.csv" )
elif [[ -x "$HR/hrocks_parse_range.sh" ]]; then
  ( cd "$HR" && SIZES="$SIZES" ./hrocks_parse_range.sh output_range "$OUT/hrocks_range.csv" )
else
  echo "[H-Rocks] range parser not found; expected parse_range.sh or hrocks_parse_range.sh"
fi

# pmem-rocksdb
if [[ -x "$PM/pmem_parse_range.sh" ]]; then
  ( cd "$PM" && SIZES="$SIZES" ./pmem_parse_range.sh output_range "$OUT/pmem_range.csv" )
else
  echo "[pmem-rocksdb] pmem_parse_range.sh not found; skipping parse."
fi

# Plush
if [[ -x "$PL/parse_range.sh" ]]; then
  ( cd "$PL" && SIZES="$SIZES" ./parse_range.sh output_range "$OUT/plush_range.csv" )
else
  echo "[Plush] parse_range.sh not found; skipping parse."
fi

################
# 3) Make plot #
################
python3 "$ROOT/scripts/plot_lines_from_csvs.py" \
  --title "Figure 9(a): RANGE query — Sustained Throughput vs Arrival Rate" \
  --xlabel "Arrival request rate (ops/s)" \
  --ylabel "Sustained Throughput (ops/s)" \
  --out "$OUT/fig9a_range.pdf" \
  $( [[ -f "$OUT/hrocks_range.csv" ]] && echo --series "$OUT/hrocks_range.csv:H-Rocks" ) \
  $( [[ -f "$OUT/pmem_range.csv"   ]] && echo --series "$OUT/pmem_range.csv:RocksDB" ) \
  $( [[ -f "$OUT/plush_range.csv"  ]] && echo --series "$OUT/plush_range.csv:Plush" )

echo "Done: $OUT"
