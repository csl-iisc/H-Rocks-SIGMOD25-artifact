#!/bin/bash

NKEYS=100000000
NTHREADS="1 2 4 8 16 32 64 128"

mkdir -p output_diff_threads_put/
cd build/
cmake .. && make -j 10
cd ../

for thread in $NTHREADS; do
    rm -rf /pmem/*
    echo $thread
    ./build/prefill_put -n $NKEYS -k 8 -v 8 -t $thread > output_diff_threads_put/output_100M_8_8_${thread}
done
