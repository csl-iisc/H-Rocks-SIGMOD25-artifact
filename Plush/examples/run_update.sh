#!/bin/bash

SIZES="500 1000 2000 4000 8000 10000 16000 25000 50000 100000 200000 400000 800000 1000000 2000000 4000000 8000000 10000000"
thread=128
PREFILL_SIZE=50000000

mkdir -p ../output_updates/
cd ../build/
cmake .. && make -j 10
cd ../

for size in $SIZES; do
    echo $size
    rm -rf /pmem/plush_table
    mkdir -p /pmem/plush_table
    ./build/test_updates -p $PREFILL_SIZE -u $size -t $thread > output_updates/output_${size}_${thread}
done
