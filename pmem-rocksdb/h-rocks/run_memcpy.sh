#!/bin/bash

VAL_SIZES="8 16 32 64 128 256 512 1024"
NTHREADS="8"
NKEYS="1000000"

make lib
make bin/memcpy_experiment
mkdir -p output_memcpy_experiment_persisted/
for keys in $NKEYS; do
    for val_size in $VAL_SIZES; do
        for thread in $NTHREADS; do
            echo $val_size $thread
            ./bin/memcpy_experiment -n $keys -t $thread -v $val_size > output_memcpy_experiment_persisted/output_${keys}_${val_size}_${thread}
            sleep 1
        done
    done
done
