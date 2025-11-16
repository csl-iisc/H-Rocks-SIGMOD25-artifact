#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
OUT="$ROOT/out/fig9b"; mkdir -p "$OUT"

HR="$ROOT/pmem-rocksdb/h-rocks"
PM="$ROOT/pmem-rocksdb/benchmarks"

echo "[Figure 9b] output dir: $OUT"

############################
# 1) (optional) run sweep  #
############################
if [[ -x "$HR/run_atomic_txn.sh" ]]; then
  (cd "$HR" && ./run_atomic_txn.sh)
else
  echo "[H-Rocks] run_atomic_txn.sh not found; skipping run."
fi

if [[ -x "$PM/run_atomic_txn.sh" ]]; then
  (cd "$PM" && ./run_atomic_txn.sh)
else
  echo "[pmem-rocksdb] run_atomic_txn.sh not found; skipping RocksDB run."
fi

#########################################
# 2) Parse logs -> CSVs for each system #
#########################################
SERIES_ARGS=()

if [[ -x "$HR/parse_atomic_txn.sh" ]]; then
  (cd "$HR" && ./parse_atomic_txn.sh output_atomic_txn "$OUT/hrocks_atomic_txn.csv")
  SERIES_ARGS+=( "--series" "$OUT/hrocks_atomic_txn.csv:H-Rocks" )
else
  echo "[H-Rocks] parse_atomic_txn.sh not found; skipping parse."
fi

if [[ -x "$PM/parse_atomic_txn.sh" ]]; then
  (cd "$PM" && ./parse_atomic_txn.sh output_atomic_txn "$OUT/pmem_atomic_txn.csv")
  SERIES_ARGS+=( "--series" "$OUT/pmem_atomic_txn.csv:RocksDB" )
else
  echo "[pmem-rocksdb] parse_atomic_txn.sh not found; skipping parse."
fi

if [[ ${#SERIES_ARGS[@]} -eq 0 ]]; then
  echo "No CSVs produced; nothing to plot." >&2
  exit 0
fi

################
# 3) Make plot #
################
python3 "$ROOT/scripts/plot_lines_from_csvs.py" \
  --title "Figure 9(b): Atomic transactions" \
  --xlabel "Transaction sizes" \
  --ylabel "Throughput (Mops/sec)" \
  --y-mops \
  --out "$OUT/fig9b_atomic_txn.pdf" \
  "${SERIES_ARGS[@]}"

echo "Done: $OUT"
