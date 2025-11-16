#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

# Allow overriding via environment while keeping the paper defaults.
PREFILL_SIZE="${PREFILL_SIZE:-10000000}"
SIZES="${SIZES:-10000 50000 100000 500000 1000000 5000000 10000000}"

make lib
make bin/test_atomic_txn
mkdir -p output_atomic_txn
for size in $SIZES; do
    hr_rm_rf "${HR_PMEM_DIR:?}/"* "${HR_SHM_DIR:?}/"*
    echo "batch_size=$size"
    # Capture wall time to derive throughput later.
    /usr/bin/time -f "elapsed_s %e" ./bin/test_atomic_txn -n "$PREFILL_SIZE" -b "$size" -k 8 -v 8 \
        > "output_atomic_txn/output_${PREFILL_SIZE}_${size}" 2>&1
done
