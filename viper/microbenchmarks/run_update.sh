PREFILL_SIZE="5000000"

SIZES="500 1000 2000 4000 8000 10000 16000 25000 50000 100000 200000 400000 800000 1000000 2000000 4000000 8000000 10000000 20000000"

mkdir -p output_prefill_update/
cd build/
cmake .. && make -j 10
cd ../

for size in $SIZES; do
    echo $size
    rm -rf /pmem/*
    ./build/prefill_update -n $PREFILL_SIZE -u $size -t 128 > output_prefill_update/output_50M_128_$size
    sleep 2
done
