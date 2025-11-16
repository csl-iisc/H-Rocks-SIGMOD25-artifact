#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

VAL_SIZES="${VAL_SIZES:-8 16 32 64 128 256 512}"
KEY_SIZES="${KEY_SIZES:-8}"
# Figure 11 uses 10M ops; default to 10M and allow override via NKEYS env.
NKEYS="${NKEYS:-10000000}"

make lib
make bin/test_puts
OUT_DIR="${OUT_DIR:-output_put_values}"
mkdir -p "$OUT_DIR"
for keys in $NKEYS; do
  for key_size in $KEY_SIZES; do
    for val_size in $VAL_SIZES; do
      hr_rm_rf "${HR_PMEM_DIR:?}/"*
      hr_rm_rf "${HR_SHM_DIR:?}/"*
      echo "$key_size $val_size"
      ./bin/test_puts -n "$keys" -k "$key_size" -v "$val_size" > "$OUT_DIR/output_${keys}_${key_size}_${val_size}"
      sleep 1
    done
  done
done
