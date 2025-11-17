#!/bin/bash

# Add prefill first to this

SIZES="1000 5000 10000 50000 100000 500000 1000000 5000000 10000000 50000000 75000000 100000000"
VAL_SIZE=8

mkdir -p output_put_latency/
cd build/
cmake .. && make -j 10
cd ../

for size in $SIZES; do
    echo $size
    rm -rf /pmem/*
    ./build/put_latency -n $size -k 8 -v 8 > output_put_latency/output_8_100_$size
done
