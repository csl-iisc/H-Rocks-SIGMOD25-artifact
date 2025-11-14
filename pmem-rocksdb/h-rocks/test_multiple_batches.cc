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
#include "batch.h"
#include "pmem_paths.h"



#define TIME_NOW std::chrono::high_resolution_clock::now()

typedef std::vector<char> char_array;

char_array charset()
{
    //Change this to suit
    return 
        char_array({'A','B','C','D','E','F',
                'G','H','I','J','K',
                'L','M','N','O','P',
                'Q','R','S','T','U',
                'V','W','X','Y','Z',
                });
};

std::string generate_random_string(size_t length, std::function<char(void)> rand_char)
{
    std::string str(length,0);
    std::generate_n(str.begin(), length, rand_char);
    return str;
}

using namespace ROCKSDB_NAMESPACE;
int main(int argc, char **argv) 
{
    int option_char;
    uint64_t num_gets, num_puts;
    size_t key_size, value_size, batch_size;
    while ((option_char = getopt (argc, argv, ":n:k:v:b:")) != -1) {
        switch (option_char)
        {
            case 'n': num_puts = atoi (optarg); break;
            case 'k': key_size = atoi (optarg); break;
            case 'v': value_size = atoi (optarg); break;
            case 'b': batch_size = atoi (optarg); break;
            case ':': fprintf (stderr, "option needs a value\n");
            case '?': fprintf (stderr, "usage: %s [-n <number of keys>] [-k <key size>] [-v     <value size>]\n", argv[0]);
        }
    }
    uint64_t num_ops = num_puts + num_gets; 

    std::cout << "Number of puts: " << num_puts << std::endl;
    std::cout << "Number of gets: " << num_gets << std::endl;
    std::cout << "Key size: " << key_size << std::endl;
    std::cout << "Value size: " << value_size << std::endl;

    std::vector<std::string> keys(num_puts); 
    std::vector<std::string> values(num_puts); 
    const auto ch_set = charset();
    std::default_random_engine rng(std::random_device{}());
    std::uniform_int_distribution<> dist(0, ch_set.size()-1);
    auto randchar = [ch_set, &dist, &rng](){return ch_set[dist(rng)];};

    rocksdb::DB* db;
    rocksdb::Options options;
    rocksdb::WriteOptions write_options; 
    options.IncreaseParallelism(10);
    options.OptimizeLevelStyleCompaction();
    options.create_if_missing = true;
    rocksdb::Status s = rocksdb::DB::Open(options, hrocks::PmemPath("rocksdb_put"), &db);
    assert(s.ok());
    std::cout << "DB opened." << std::endl;


    for(uint64_t i = 0; i < num_puts; ++i) {
        keys[i] = generate_random_string(key_size - 1, randchar); 
        values[i] = generate_random_string(value_size - 1, randchar); 
    }

    std::vector<Command> readCommands, writeCommands, updateCommands;
    Batch batch(readCommands, writeCommands, updateCommands, 0, db);
    // Perform a put operation
    char *key, *value; 
    key = (char*)malloc(key_size); 
    value = (char*)malloc(value_size); 
    uint64_t num_batches = num_puts/batch_size; 
    for(uint64_t j = 0; j < num_batches; ++j) {
    for(uint64_t i = 0; i < batch_size; ++i) {
        strcpy(key, keys[j * num_batches + i].c_str()); 
        strcpy(value, values[j * num_batches + i].c_str()); 
        batch.Put(key, value); 
        //std::cout << key << " " << value << "\n"; 
    }
    batch.Exit(); 
    }
    return 0;
}
