#!/bin/bash

PUT_SIZES="500 1000 2000 4000 8000 10000 16000 25000 50000 100000 200000 400000 800000 1000000 2000000 4000000 8000000 10000000 20000000 40000000 50000000 80000000 100000000"
VAL_SIZES="100"
KEY_SIZE="16"

mkdir -p output_ycsbD
make lib && make bin/test_ycsbB
for val_size in $VAL_SIZES; do
    for size in $PUT_SIZES; do
        rm -rf /pmem/*
        rm -rf /dev/shm/*
        echo $size
        ./bin/test_ycsbB -n $size -k ${KEY_SIZE} -v $val_size > output_ycsbD/output_${size}_${KEY_SIZE}_${val_size}
        sleep 1
    done
done

