#include "batch.h"
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

char_array charset2()
{
    //Change this to suit
    return 
        char_array({'1','2','3','4','5','6',
                '7','8','9','0',
                });
};




std::string generate_random_string(size_t length, std::function<char(void)> rand_char)
{
    std::string str(length,0);
    std::generate_n(str.begin(), length, rand_char);
    return str;
}


int main(int argc, char **argv) 
{
    int option_char;
    uint64_t num_updates, num_puts;
    size_t key_size, value_size;
    while ((option_char = getopt (argc, argv, ":p:n:k:v:")) != -1) {
        switch (option_char)
        {
            case 'p': num_puts = atoi (optarg); break;
            case 'n': num_updates = atoi (optarg); break;
            case 'k': key_size = atoi (optarg); break;
            case 'v': value_size = atoi (optarg); break;
            case ':': fprintf (stderr, "option needs a value\n");
            case '?': fprintf (stderr, "usage: %s [-n <number of keys>] [-k <key size>] [-v     <value size>]\n", argv[0]);
        }
    }

    std::cout<<"Number of puts: " << num_puts << std::endl;
    std::cout<<"Number of updates: " << num_updates << std::endl;
    std::cout<<"Key size: " << key_size << std::endl;
    std::cout<<"Value size: " << value_size << std::endl;

    std::vector<std::string> keys(num_puts); 
    std::vector<std::string> values(num_puts); 
    const auto ch_set = charset();
    const auto ch_set2 = charset2();
    std::default_random_engine rng(std::random_device{}());
    std::uniform_int_distribution<> dist(0, ch_set.size()-1);
    std::uniform_int_distribution<> dist2(0, ch_set2.size()-1);
    auto randchar = [ch_set, &dist, &rng](){return ch_set[dist(rng)];};
    auto randchar2 = [ch_set2, &dist2, &rng](){return ch_set2[dist2(rng)];};
    

    for(uint64_t i = 0; i < num_puts; ++i) {
        keys[i] = generate_random_string(key_size - 1, randchar); 
        values[i] = generate_random_string(value_size - 1, randchar2); 
    }

    rocksdb::DB* db;
    rocksdb::Options options;
    rocksdb::WriteOptions write_options; 
    options.IncreaseParallelism(10);
    options.OptimizeLevelStyleCompaction();
    options.create_if_missing = true;
    rocksdb::Status s = rocksdb::DB::Open(options, "/pmem/rocksdb_update", &db);
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
       // Perform a get operation
    for(uint64_t i = 0; i < num_updates; ++i) {
        uint64_t get_index = rand() % num_puts; 
        strcpy(key, keys[get_index].c_str()); 
        //std::cout << key << " " << value << "\n"; 
        //std::cout << key << "\n"; 
        batch.Merge(key); 
    }
  batch.Exit(); 
}
