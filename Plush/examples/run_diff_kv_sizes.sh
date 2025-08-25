#!/bin/bash

VAL_SIZES="8 16 32 64 100 128 256 512 1024"
KEY_SIZES="8 16 32 64 100"
NKEYS="1000000 5000000"

mkdir -p output_diff_kv_sizes/
cd build/
cmake .. && make -j 10
cd ../

for keys in $NKEYS; do
    for key_size in $KEY_SIZES; do
        for val_size in $VAL_SIZES; do
            rm -rf /pmem/*
            echo $key_size $val_size
            ./build/test_put_mt -n $keys -k $key_size -v $val_size -t 128 > output_diff_kv_sizes/output_5M_${key_size}_${val_size}
        done
    done
done
