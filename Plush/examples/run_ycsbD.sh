#!/bin/bash

SIZES="10000 100000 500000 1000000 10000000 25000000 50000000 75000000 100000000"

val_size=100
thread=32
PREFILL_SIZE="${PREFILL_SIZE:-50000000}"

mkdir -p ../output_ycsbD/
cd ../build/
cmake .. && make -j 10
cd ../

for size in $SIZES; do
    echo $size
    rm -rf /pmem/plush_table
    mkdir -p /pmem/plush_table
    ./build/test_ycsbD -p $PREFILL_SIZE -n $size -k 16 -v $val_size -t $thread > output_ycsbD/output_${size}_16_${val_size}_${thread}
done
