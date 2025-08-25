#!/bin/bash


rm out.txt
rm /pmem/fptree
rm /pmem/pmem_hash.data
rm /pmem/pmem_hash_var.data
rm -rf /pmem/viper/
rm /pmem/fastfair
rm /pmem/pmemkv
rm -rf /pmem/rocksdb/
rm /pmem/utree
rm -rf /pmem/tabletest
rm /pmem/dptree.dat

mkdir /pmem/tabletest
