#!/bin/bash

#SIZES="500 1000 2000 4000 8000 10000 16000 25000 50000 100000 200000 400000 800000 1000000 2000000 4000000 5000000 10000000 50000000"
#NKEYS=50000000
SIZES="500 1000 2000 4000 8000 10000 16000 25000 50000 100000 200000 400000 800000 1000000 2000000 4000000 8000000 10000000"
NKEYS="50000000"

mkdir -p output_range_updated2/
make lib
make bin/test_range

#for i in 0 1 2 3 4; do
for size in $SIZES; do
    rm -rf /pmem/* && rm -rf /dev/shm/*
    echo $size
    ./bin/test_range -p $NKEYS -g $size -k 8 -v 8 > output_range_updated2/output_50M_$size
done
#done
