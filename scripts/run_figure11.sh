#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
OUT="$ROOT/out/fig11"; mkdir -p "$OUT"

HR="$ROOT/pmem-rocksdb/h-rocks"

echo "[Figure 11] output dir: $OUT"

# Baseline (value pointers disabled)
( cd "$HR" && OUT_DIR="output_put_values_base" HR_PUTS_WITH_VALUES=0 ./run_puts_value.sh )
( cd "$HR" && OUT_DIR="output_get_values_base" HR_PUTS_WITH_VALUES=0 ./run_gets_value.sh )

# With values enabled via env toggle
( cd "$HR" && OUT_DIR="output_put_values_with_values" HR_PUTS_WITH_VALUES=1 ./run_puts_value.sh )
( cd "$HR" && OUT_DIR="output_get_values_with_values" HR_PUTS_WITH_VALUES=1 ./run_gets_value.sh )

# Parse to CSVs
( cd "$HR" && ./parse_put_values.sh output_put_values_base "$OUT/hrocks_puts_base.csv" )
( cd "$HR" && ./parse_put_values.sh output_put_values_with_values "$OUT/hrocks_puts_with_values.csv" )
( cd "$HR" && ./parse_get_values.sh  output_get_values_base  "$OUT/hrocks_gets_base.csv" )
( cd "$HR" && ./parse_get_values.sh  output_get_values_with_values "$OUT/hrocks_gets_with_values.csv" )

# Plot both panels
python3 "$ROOT/scripts/plot_fig11.py" \
  --puts-base "$OUT/hrocks_puts_base.csv" \
  --puts-with-values "$OUT/hrocks_puts_with_values.csv" \
  --gets-base "$OUT/hrocks_gets_base.csv" \
  --gets-with-values "$OUT/hrocks_gets_with_values.csv" \
  --out "$OUT/fig11.png"

echo "Baseline logs:      $HR/output_put_values_base  and $HR/output_get_values_base"
echo "With-values logs:   $HR/output_put_values_with_values  and $HR/output_get_values_with_values"
echo "Plot:               $OUT/fig11.png"
