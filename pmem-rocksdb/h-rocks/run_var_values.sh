#!/bin/bash

VAL_SIZES="8 16 32 64 128 256 512"
KEY_SIZES="8"
#NKEYS="1000000 5000000 10000000 50000000"

make lib
make bin/test_get_put
mkdir -p output_diff_values_valuePtr/
for key_size in $KEY_SIZES; do
    for val_size in $VAL_SIZES; do
        rm -rf /pmem/*
        rm -rf /dev/shm/*
        echo $key_size $val_size
        ./bin/test_get_put -p 5000000 -g 5000000 -k $key_size -v $val_size > output_diff_values_valuePtr/output_5M_5M_${key_size}_${val_size}
        sleep 1
        ./bin/test_get_put -p 1000000 -g 9000000 -k $key_size -v $val_size > output_diff_values_valuePtr/output_1M_9M_${key_size}_${val_size}
        sleep 1
    done
done
