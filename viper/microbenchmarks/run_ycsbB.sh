#!/bin/bash

# Add prefill first to this

SIZES="10000 25000 50000 100000 500000 1000000 500000 10000000 20000000 50000000 100000000"

mkdir -p output_ycsbB4/
cd build/
cmake .. && make -j 10
cd ../

for size in $SIZES; do
    echo $size
    rm -rf /pmem/*
    ./build/ycsbB -n $size -k 16 -v 100 -t 128 > output_ycsbB4/output_16_100_$size
done
