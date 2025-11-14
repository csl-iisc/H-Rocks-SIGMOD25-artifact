#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

# Sizes & params
SIZES="10000 1000000 10000000 25000000 50000000 75000000 100000000"
VAL_SIZES="8"
PREFILL_SIZE="50000000"
KEY_SIZE="8"

OUT_DIR="output_updates"
DB_PATH="$HR_PMEM_DIR/rocksdb_update"

mkdir -p "$OUT_DIR"

# Build once
make clean && make lib
make bin/test_updates

for v in $VAL_SIZES; do
  for n in $SIZES; do
    echo "==> updates: prefill=$PREFILL_SIZE ops=$n k=$KEY_SIZE v=$v"
    hr_rm_rf "$DB_PATH" "${DB_PATH}_"* \
      "${HR_PMEM_DIR:?}"/rocksdb_* "${HR_PMEM_DIR:?}"/values* "${HR_PMEM_DIR:?}"/hrocks_* \
      "${HR_SHM_DIR:?}"/rocksdb_* "${HR_SHM_DIR:?}"/values* "${HR_SHM_DIR:?}"/hrocks_*
    ./bin/test_updates -p "$PREFILL_SIZE" -n "$n" -k "$KEY_SIZE" -v "$v" \
      > "${OUT_DIR}/output_${KEY_SIZE}_${v}_${n}"
    sleep 2
  done
done
