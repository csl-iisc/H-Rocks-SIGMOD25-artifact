#!/bin/bash

# Add prefill first to this

SIZES="500 1000 2000 4000 8000 10000 16000 25000 50000 100000 200000 400000 800000 1000000 2000000 4000000 8000000 10000000 20000000 40000000 50000000 80000000 100000000"
#VAL_SIZES="8 100"
VAL_SIZES=8

mkdir -p output_puts9/
make clean && make lib
make bin/test_puts

for val_size in $VAL_SIZES; do
    for size in $SIZES; do
        echo $size
        rm -rf /pmem/rocksdb_puts
        rm -rf /pmem/values*
        rm -rf /dev/shm/*
        ./bin/test_puts -n $size -k 8 -v $val_size > output_puts9/output_8_${val_size}_$size
        sleep 2
    done
done
