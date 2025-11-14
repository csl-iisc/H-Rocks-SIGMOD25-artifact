#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

PREFILL_SIZE="10000000"
SIZES="10000 50000 100000 500000 1000000 5000000 10000000"

make lib
make bin/test_atomic_txn
mkdir -p output_atomic_txn2
for size in $SIZES; do
    hr_rm_rf "${HR_PMEM_DIR:?}/"*
    echo $size
    ./bin/test_atomic_txn -n $PREFILL_SIZE -b $size -k 8 -v 8 > output_atomic_txn2/output_10M_$size
done
