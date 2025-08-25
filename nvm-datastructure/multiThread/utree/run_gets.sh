#!/bin/bash

PREFILL_SIZE=50000000

SIZES="500 1000 2000 4000 8000 10000 16000 25000 50000 100000 200000 400000 800000 1000000       2000000 4000000 8000000 10000000 20000000 40000000 50000000 80000000 100000000"

mkdir output_gets/
./build.sh
for size in $SIZES; do
    echo $size
    rm -rd /pmem/utree
    ./get_test -p $PREFILL_SIZE -n $size -k 8 -v 8 -t 16 > output_gets/output_${size}_8_8
    sleep 1
done
