#!/bin/bash

# Add prefill first to this

SIZES="500 1000 2000 4000 8000 10000 16000 25000 50000 100000 200000 400000 800000 1000000 2000000 4000000 8000000 10000000 20000000 40000000 50000000 80000000 100000000"

mkdir -p output_ycsbD/
cd build/
cmake .. && make -j 10
cd ../

for size in $SIZES; do
    echo $size
    rm -rf /pmem/*
    ./build/ycsbB -n $size -k 16 -v 100 -t 128 > output_ycsbD/output_16_100_$size
done
