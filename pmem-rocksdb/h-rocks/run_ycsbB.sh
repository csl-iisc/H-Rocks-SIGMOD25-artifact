#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

SIZES="10000 1000000 10000000 25000000 50000000 75000000 100000000"
VAL_SIZES="100"
KEY_SIZE="16"

OUT_DIR="output_ycsbB"
DB_PATH="$HR_PMEM_DIR/rdb_ycsbB"   # match the path used by test_ycsbB

mkdir -p "$OUT_DIR"

make lib && make bin/test_ycsbB

for v in $VAL_SIZES; do
  for n in $SIZES; do
    echo "==> ycsbB: ops=$n k=$KEY_SIZE v=$v"
    hr_rm_rf "$DB_PATH" "${HR_SHM_DIR:?}/"*
    ./bin/test_ycsbB -n "$n" -k "$KEY_SIZE" -v "$v" \
      > "${OUT_DIR}/output_${n}_${KEY_SIZE}_${v}"
    sleep 1
  done
done
