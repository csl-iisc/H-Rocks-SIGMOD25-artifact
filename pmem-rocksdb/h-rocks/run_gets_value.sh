#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

VAL_SIZES="${VAL_SIZES:-8 16 32 64 128 256 512}"
KEY_SIZE="${KEY_SIZE:-8}"
NGETS="${NGETS:-10000000}"          # total GETs (prefill is fixed in test_get_put)
PREFILL="${PREFILL:-10000000}"
OUT_DIR="${OUT_DIR:-output_get_values}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-1}"

mkdir -p "$OUT_DIR"
make lib
make bin/test_puts bin/test_get_put

for val_size in $VAL_SIZES; do
  echo "GET sweep: n=$NGETS k=$KEY_SIZE v=$val_size HR_PUTS_WITH_VALUES=${HR_PUTS_WITH_VALUES:-0}"
  hr_rm_rf "${HR_PMEM_DIR:?}"/rocksdb_* "${HR_PMEM_DIR:?}"/values* \
           "${HR_SHM_DIR:?}"/rocksdb_* "${HR_SHM_DIR:?}"/values* "${HR_SHM_DIR:?}"/hrocks_*
  ./bin/test_get_put -p "$PREFILL" -g "$NGETS" -k "$KEY_SIZE" -v "$val_size" \
    > "$OUT_DIR/get_put_k${KEY_SIZE}_v${val_size}_n${NGETS}.log"
  sleep "$SLEEP_BETWEEN"
done
