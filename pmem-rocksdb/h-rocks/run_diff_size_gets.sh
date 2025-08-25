#!/bin/bash

VAL_SIZES="8 16 32 64 128 256 512 1024"
KEY_SIZES="8 16 32 64 128"
#NKEYS="1000000 5000000 10000000 50000000"
NKEYS="1000000 10000000"
PREFILL=10000000

make lib
make bin/test_get_put
mkdir -p output_diff_sizes_gets/
for keys in $NKEYS; do
    for key_size in $KEY_SIZES; do
        for val_size in $VAL_SIZES; do
            rm -rf /pmem/*
            rm -rf /dev/shm/*
            echo $key_size $val_size
            ./bin/test_get_put -p $PREFILL -g $keys -k $key_size -v $val_size > output_diff_sizes_gets/output_${keys}_${key_size}_${val_size}
            sleep 1
        done
    done
done
