#include "batch.h"
#include "pmem_paths.h"
#include <iostream> 
#include <cstdio>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <functional>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <getopt.h>
#include <unistd.h>
#include <algorithm>  //for std::generate_n
#include <set>

#include <bits/stdc++.h>
#include <rocksdb/db.h>
#include <rocksdb/slice.h>
#include <rocksdb/options.h>
#include <rocksdb/write_batch.h>
#include "rocksdb/statistics.h"


#define TIME_NOW std::chrono::high_resolution_clock::now()

using namespace rocksdb;
int main(int argc, char **argv) 
{
    int option_char;
    uint64_t num_gets, num_puts;
    size_t key_size, value_size;
    while ((option_char = getopt (argc, argv, ":g:p:k:v:")) != -1) {
        switch (option_char)
        {
            case 'g': num_gets = atoi (optarg); break;
            case 'p': num_puts = atoi (optarg); break;
            case ':': fprintf (stderr, "option needs a value\n");
            case '?': fprintf (stderr, "usage: %s [-n <number of keys>] [-k <key size>] [-v     <value size>]\n", argv[0]);
        }
    }
    uint64_t num_ops = num_puts + num_gets; 

    std::cout<<"Number of puts: " << num_puts << std::endl;
    std::cout<<"Number of gets: " << num_gets << std::endl;
    std::cout<<"Key size: " << key_size << std::endl;
    std::cout<<"Value size: " << value_size << std::endl;

    std::vector<std::string> keys(num_puts); 
    std::vector<std::string> values(num_puts); 

    rocksdb::DB* db;
    rocksdb::Options options;
    rocksdb::WriteOptions write_options; 
    options.IncreaseParallelism(10);
    options.OptimizeLevelStyleCompaction();
    options.create_if_missing = true;
    rocksdb::Status s = rocksdb::DB::Open(options, hrocks::PmemPath("rocksdb_get"), &db);
    assert(s.ok());
    std::cout << "DB opened." << std::endl;


    std::vector<Command> readCommands, writeCommands, updateCommands;
    Batch batch(readCommands, writeCommands, updateCommands, 0, db);
    // Perform a put operation
    for(uint64_t i = 0; i < num_puts; ++i) {
        const char* key = "sameKey"; 
        std::string value = "val_" + std::to_string(100 + i);
        const char* charValue = value.c_str();
        batch.Put(key, charValue); 
        batch.Put(key, charValue); 
        batch.Get(key); 
    }

    // Perform a get operation
    batch.Exit(); 
}
