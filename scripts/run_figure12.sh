#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
OUT="$ROOT/out/fig12_latency"; mkdir -p "$OUT"

HR="$ROOT/pmem-rocksdb/h-rocks"
PM="$ROOT/pmem-rocksdb/examples"

echo "[Fig12] output dir: $OUT"

# 1) (Re)parse to build throughput CSVs (we only need H-Rocks + RocksDB)
SIZES="${SIZES:-}"

# PUTS
( cd "$HR" && SIZES="$SIZES" ./parse_puts.sh   output_puts   "$OUT/hrocks_puts.csv" )
( cd "$PM" && SIZES="$SIZES" ./pmem_parse_puts.sh output_puts "$OUT/pmem_puts.csv" )

# GETS
( cd "$HR" && SIZES="$SIZES" ./parse_gets.sh   .             "$OUT/hrocks_gets.csv" )
( cd "$PM" && SIZES="$SIZES" ./pmem_parse_gets.sh output_gets "$OUT/pmem_gets.csv" )

# 2) Plot latency (ms). Use log-x to match the figure look. Y-limits from your panel.
python3 "$ROOT/scripts/plot_latency_from_csvs.py" \
  --title "PUT latencies with varying request rate" \
  --xlabel "Arrival request rate (ops/sec)" \
  --ylabel "Latency (msecs)" \
  --xlog \
  --ylim 0 3000 \
  --out "$OUT/fig12_puts_latency.png" \
  --series "$OUT/hrocks_puts.csv:H-Rocks" \
  --series "$OUT/pmem_puts.csv:RocksDB"

python3 "$ROOT/scripts/plot_latency_from_csvs.py" \
  --title "GET latencies with varying request rate" \
  --xlabel "Arrival request rate (ops/sec)" \
  --ylabel "Latency (msecs)" \
  --xlog \
  --ylim 0 900 \
  --out "$OUT/fig12_gets_latency.png" \
  --series "$OUT/hrocks_gets.csv:H-Rocks" \
  --series "$OUT/pmem_gets.csv:RocksDB"

echo "Done: $OUT"
