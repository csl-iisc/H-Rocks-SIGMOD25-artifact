#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
OUT="$ROOT/out/fig10"; mkdir -p "$OUT"

HR="$ROOT/pmem-rocksdb/h-rocks"
PM="$ROOT/pmem-rocksdb/benchmarks"
VP="$ROOT/viper/microbenchmarks"
PL="$ROOT/Plush/examples"

echo "[Figure 10] output dir: $OUT"

# 1) (optional) run variable-KV experiments if runners exist
[[ -x "$HR/run_var_kv_puts.sh" ]] && (cd "$HR" && ./run_var_kv_puts.sh) || echo "[H-Rocks] run_var_kv_puts.sh not found, skipping."
[[ -x "$HR/run_var_kv_gets.sh" ]] && (cd "$HR" && ./run_var_kv_gets.sh) || echo "[H-Rocks] run_var_kv_gets.sh not found, skipping."

[[ -x "$PM/run_var_kv_puts.sh" ]] && (cd "$PM" && ./run_var_kv_puts.sh) || echo "[pmem-rocksdb] run_var_kv_puts.sh not found, skipping."
[[ -x "$PM/run_var_kv_gets.sh" ]] && (cd "$PM" && ./run_var_kv_gets.sh) || echo "[pmem-rocksdb] run_var_kv_gets.sh not found, skipping."

[[ -x "$VP/run_var_kv_puts.sh" ]] && (cd "$VP" && ./run_var_kv_puts.sh) || echo "[Viper] run_var_kv_puts.sh not found, skipping."
[[ -x "$VP/run_var_kv_gets.sh" ]] && (cd "$VP" && ./run_var_kv_gets.sh) || echo "[Viper] run_var_kv_gets.sh not found, skipping."

[[ -x "$PL/run_var_kv_puts.sh" ]] && (cd "$PL" && ./run_var_kv_puts.sh) || echo "[Plush] run_var_kv_puts.sh not found, skipping."
[[ -x "$PL/run_var_kv_gets.sh" ]] && (cd "$PL" && ./run_var_kv_gets.sh) || echo "[Plush] run_var_kv_gets.sh not found, skipping."

# 2) Parse -> CSVs (expect ops/sec; plotter converts to Mops/s)
PUTS_ARGS=()
GETS_ARGS=()

# H-Rocks (if you have parsers)
if [[ -x "$HR/parse_kv_puts.sh" ]]; then
  (cd "$HR" && ./parse_kv_puts.sh output_diff_sizes_values "$OUT/hrocks_puts.csv")
  PUTS_ARGS+=( "--puts" "$OUT/hrocks_puts.csv:H-Rocks" )
fi
if [[ -x "$HR/parse_kv_gets.sh" ]]; then
  (cd "$HR" && ./parse_kv_gets.sh output_diff_sizes_values "$OUT/hrocks_gets.csv")
  GETS_ARGS+=( "--gets" "$OUT/hrocks_gets.csv:H-Rocks" )
fi

# pmem-rocksdb
if [[ -x "$PM/parse_var_kv_puts.sh" ]]; then
  (cd "$PM" && ./parse_var_kv_puts.sh output_var_kv_puts "$OUT/pmem_puts.csv")
  PUTS_ARGS+=( "--puts" "$OUT/pmem_puts.csv:RocksDB" )
fi
if [[ -x "$PM/parse_var_kv_gets.sh" ]]; then
  (cd "$PM" && ./parse_var_kv_gets.sh output_var_kv_gets "$OUT/pmem_gets.csv")
  GETS_ARGS+=( "--gets" "$OUT/pmem_gets.csv:RocksDB" )
fi

# Viper (parsers optional—add if present)
if [[ -x "$VP/parse_var_kv_puts.sh" ]]; then
  (cd "$VP" && ./parse_var_kv_puts.sh output_var_kv_puts "$OUT/viper_puts.csv")
  PUTS_ARGS+=( "--puts" "$OUT/viper_puts.csv:Viper" )
fi
if [[ -x "$VP/parse_var_kv_gets.sh" ]]; then
  (cd "$VP" && ./parse_var_kv_gets.sh output_var_kv_gets "$OUT/viper_gets.csv")
  GETS_ARGS+=( "--gets" "$OUT/viper_gets.csv:Viper" )
fi

# Plush
if [[ -f "$PL/parse_var_kv_puts.sh" ]]; then
  (cd "$PL" && bash ./parse_var_kv_puts.sh output_var_kv_puts "$OUT/plush_puts.csv")
  PUTS_ARGS+=( "--puts" "$OUT/plush_puts.csv:Plush" )
else
  echo "[Plush] parse_var_kv_puts.sh not found, skipping."
fi
if [[ -f "$PL/parse_var_kv_gets.sh" ]]; then
  (cd "$PL" && bash ./parse_var_kv_gets.sh output_var_kv_gets "$OUT/plush_gets.csv")
  GETS_ARGS+=( "--gets" "$OUT/plush_gets.csv:Plush" )
else
  echo "[Plush] parse_var_kv_gets.sh not found, skipping."
fi

# 3) Plot two panels (PUTs / GETs) over kv = {8/8,16/32,16/128,32/256,64/128,128/1024}
python3 "$ROOT/scripts/run_kv_mops.py" \
  --title_puts "PUTs with varying key-value sizes" \
  --title_gets "GETs with varying key-value sizes" \
  --out_dir "$OUT" \
  "${PUTS_ARGS[@]}" \
  "${GETS_ARGS[@]}"

echo "[Figure 10] Done: $OUT"
