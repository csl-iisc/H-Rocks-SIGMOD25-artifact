#!/bin/bash

# Add prefill first to this

SIZES="500 1000 2000 4000 8000 10000 16000 25000 50000 100000 200000 400000 800000 1000000 2000000 4000000 8000000 10000000 20000000 40000000 50000000 80000000 100000000"
#VAL_SIZES="8 100"
VAL_SIZES=8
PREFILL_SIZE=50000000

mkdir -p output_gets4/
make clean && make lib
make bin/test_get_put

for val_size in $VAL_SIZES; do
    for size in $SIZES; do
        echo $size
        rm -rf /pmem/rocksdb_gets
        rm -rf /pmem/values*
        rm -rf /dev/shm/*
        ./bin/test_get_put -p $PREFILL_SIZE -g $size -k 8 -v $val_size > output_gets4/output_50M_8_${val_size}_$size
        sleep 2
    done
done
