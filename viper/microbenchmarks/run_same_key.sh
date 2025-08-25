#!/bin/bash

ITER="1 2 3 4 5 6 7 8 9 10"

for i in $ITER; do
    echo $key_size $val_size
    ./build/prefill_put -n $keys -k $key_size -v $val_size -t 128 > output_diff_sizes/output_5M_${key_size}_${val_size}
done
