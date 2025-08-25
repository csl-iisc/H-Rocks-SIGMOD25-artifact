#!/bin/bash

NKEYS=100000000
NTHREADS="128 64 32 16"
VAL_SIZES="8 100"

mkdir -p ../output_diff_threads_puts/
cd ../build/
cmake .. && make -j 10
cd ../

for thread in $NTHREADS; do
    echo $thread
    rm -rf /pmem/plush_table
    mkdir -p /pmem/plush_table
    ./build/test_put_int -n $NKEYS -t $thread > output_diff_threads_puts/output_100M_int_${thread}
done
