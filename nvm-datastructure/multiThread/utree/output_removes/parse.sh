#!/bin/bash

SIZES="500 1000 2000 4000 8000 10000 16000 25000 50000 100000 200000 400000 800000 1000000 2000000 4000000 8000000 10000000 20000000 40000000"

for size in $SIZES; do
    insert_time=$(grep -nre insert_time output_${size}_8_8 | awk '{print $NF}')
    echo "$size $thread $insert_time"
done

