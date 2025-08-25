#!/bin/bash

SIZES="500 1000 2000 4000 8000 10000 16000 25000 50000 100000 200000 400000 800000 1000000 2000000 4000000 8000000 10000000 20000000 40000000 50000000 80000000 100000000"

val_size=8
thread=128
PREFILL_SIZE=50000000
num_queries=100

mkdir -p ../output_range/
cd ../build/
cmake .. && make -j 10
cd ../

for size in $SIZES; do
    echo $size
    rm -rf /pmem/plush_table
    mkdir -p /pmem/plush_table
    ./build/test_range -p $PREFILL_SIZE -n $size -e $num_queries -t $thread > output_range/output_${size}_8_${val_size}_${thread}
done
