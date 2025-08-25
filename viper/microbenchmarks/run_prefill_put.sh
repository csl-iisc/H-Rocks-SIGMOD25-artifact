#!/bin/bash

# Add prefill first to this

SIZES="10000 100000 200000 400000 800000 1000000 2000000 4000000 8000000 10000000 20000000 40000000 50000000 80000000 100000000"
VAL_SIZE=8

mkdir -p output_prefill_put/
mkdir -p build/
cd build/
cmake .. -DCMAKE_BUILD_TYPE=Release && make
cd ../

for size in $SIZES; do
    echo $size
    rm -rf /pmem/viper_prefill_put
    ./build/prefill_put -n $size -k 8 -v 8 -t 128 > output_prefill_put/output_8_8_$size
done
