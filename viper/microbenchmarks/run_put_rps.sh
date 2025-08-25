#!/bin/bash

# Add prefill first to this

SIZES="10000 50000 100000 500000 1000000 5000000 10000000 20000000 50000000 80000000 100000000"
VAL_SIZE="8"

mkdir -p output_rps_put2/
cd build/
cmake .. && make -j 10
cd ../

for size in $SIZES; do
for val in $VAL_SIZE; do
    echo $size
    rm -rf /pmem/viper_prefill_put
    ./build/put_rps -n $size -k 8 -v $val -t 128 > output_rps_put2/output_8_${val}_$size
done
done
