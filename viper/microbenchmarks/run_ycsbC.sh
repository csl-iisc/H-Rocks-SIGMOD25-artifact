
PREFILL_SIZE="50000000"

SIZES="10000 100000 500000 1000000 10000000 25000000 50000000 75000000 100000000"

mkdir -p output_ycsbC/
cd build/
cmake .. && make -j 10
cd ../

for size in $SIZES; do
    echo $size
    rm -rf /pmem/*
    ./build/prefill_get -p $PREFILL_SIZE -n $size -k 16 -v 100 -t 32 > output_ycsbC/output_16_100_$size
    sleep 2
done
