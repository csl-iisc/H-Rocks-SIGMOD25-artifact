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
            case 'k': key_size = atoi (optarg); break;
            case 'v': value_size = atoi (optarg); break;
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
    const auto ch_set = charset();
    std::default_random_engine rng(std::random_device{}());
    std::uniform_int_distribution<> dist(0, ch_set.size()-1);
    auto randchar = [ch_set, &dist, &rng](){return ch_set[dist(rng)];};

    for(uint64_t i = 0; i < num_puts; ++i) {
        keys[i] = generate_random_string(key_size - 1, randchar); 
        values[i] = generate_random_string(value_size - 1, randchar); 
    }

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
    char *key, *value; 
    key = (char*)malloc(key_size); 
    value = (char*)malloc(value_size); 
    for(uint64_t i = 0; i < num_puts; ++i) {
        strcpy(key, keys[i].c_str()); 
        strcpy(value, values[i].c_str()); 
        batch.Put(key, value); 
        //std::cout << key << " " << value << "\n"; 
    }

    char *key1, *key2; 
    key1 = (char*)malloc(key_size); 
    key2 = (char*)malloc(key_size); 
    // Perform a get operation
    for(uint64_t i = 0; i < num_gets; ++i) {
        uint64_t get_index1 = rand() % num_puts; 
        uint64_t get_index2 = rand() % num_puts; 
        uint64_t small_index, big_index; 
        if(get_index1 < get_index2) {
            small_index = get_index1; 
            big_index = get_index2;
        } else {
            big_index = get_index1; 
            small_index = get_index2;
        }
        strcpy(key1, keys[small_index].c_str()); 
        strcpy(key2, keys[big_index].c_str()); 
        batch.Range(key1, key2, 0); 
    }
    batch.Exit(); 
}
