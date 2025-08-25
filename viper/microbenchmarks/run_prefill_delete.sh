#!/bin/bash

PREFILL_SIZE="50000000"

SIZES="500 1000 2000 4000 8000 10000 16000 25000 50000 100000 200000 400000 800000 1000000 2000000 4000000 8000000 10000000 20000000 40000000 50000000 80000000 100000000"

#VAL_SIZES="8 100"
VAL_SIZES=8

mkdir -p output_prefill_delete2/
cd build/
cmake .. && make -j 10
cd ../

for val_size in $VAL_SIZES; do
    for size in $SIZES; do
        echo $val_size $size
        rm -rf /pmem/viper/*
        ./build/prefill_delete -p $PREFILL_SIZE -n $size -k 8 -v $val_size -t 128 > output_prefill_delete2/output_8_${val_size}_$size
        sleep 2
    done
done
