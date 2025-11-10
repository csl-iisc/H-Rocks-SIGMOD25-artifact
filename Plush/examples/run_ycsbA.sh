#!/bin/bash

SIZES="10000 100000 500000 1000000 10000000 25000000 50000000 75000000 100000000"

val_size=100
thread=32

mkdir -p ../output_ycsbA/
cd ../build/
cmake .. && make -j 10
cd ../

for size in $SIZES; do
    echo $size
    rm -rf /pmem/plush_table
    mkdir -p /pmem/plush_table
    ./build/test_ycsbA -n $size -k 16 -v $val_size -t $thread > output_ycsbA/output_${size}_16_${val_size}_${thread}
done
