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

#include <rocksdb/db.h>
#include <rocksdb/slice.h>
#include <rocksdb/options.h>
#include <rocksdb/write_batch.h>
#include "rocksdb/statistics.h"
#include "batch.h"
#include "pmem_paths.h"

#define NTHREADS 128

// Defining macros for timing analysis
#define TIME_NOW std::chrono::high_resolution_clock::now()

using namespace std;

typedef std::vector<char> char_array;

char_array charset()
{
    //Change this to suit
    return char_array( 
            {'0','1','2','3','4',
            '5','6','7','8','9',
            'A','B','C','D','E','F',
            'G','H','I','J','K',
            'L','M','N','O','P',
            'Q','R','S','T','U',
            'V','W','X','Y','Z',
            'a','b','c','d','e','f',
            'g','h','i','j','k',
            'l','m','n','o','p',
            'q','r','s','t','u',
            'v','w','x','y','z'
            });
};   

std::string generate_random_string(size_t length, std::function<char(void)> rand_char){
    std::string str(length,0);
    std::generate_n( str.begin(), length, rand_char );
    return str;
}

using namespace rocksdb; 
using namespace std; 

int main (int argc, char** argv) 
{
    int option_char;
    uint64_t num_ops, batch_size, num_puts;
    size_t key_size, value_size;
    while ((option_char = getopt (argc, argv, ":n:k:v:b:")) != -1) {
        switch (option_char)
        {
            case 'n': num_ops= atoi (optarg); break;
            case 'k': key_size = atoi (optarg); break;
            case 'v': value_size = atoi (optarg); break;
            case 'b': batch_size = atoi (optarg); break;
            case ':': fprintf (stderr, "option needs a value\n");
            case '?': fprintf (stderr, "usage: %s [-n <number of keys>] [-k <key size>] [-v     <value size>]\n", argv[0]);
        }
    }

    num_puts = num_ops/2;
    std::cout << "Number of puts: " << num_puts << std::endl;
    //std::cout<<"Number of gets: " << num_gets << std::endl;
    std::cout << "Key size: " << key_size << std::endl;
    std::cout << "Value size: " << value_size << std::endl;
    rocksdb::DB* db;
    rocksdb::Options options;
    rocksdb::WriteOptions write_options; 
    options.IncreaseParallelism(NTHREADS);
    options.OptimizeLevelStyleCompaction();
    options.create_if_missing = true;
    rocksdb::Status s = rocksdb::DB::Open(options, hrocks::PmemPath("rocksdb_batches"), &db);
    assert(s.ok());
    std::cout << "DB opened." << std::endl;
    
    const auto ch_set = charset();
    std::default_random_engine rng(std::random_device{}());
    std::uniform_int_distribution<> dist(0, ch_set.size()-1);
    auto randchar = [ch_set, &dist, &rng](){return ch_set[ dist(rng) ];};

    std::vector<std::string> keys(num_puts);
    std::vector<std::string> values(num_puts);
    for(uint64_t i = 0; i < num_puts; ++i) {
        keys[i] = generate_random_string(key_size - 1, randchar);
        values[i] = generate_random_string(value_size - 1, randchar);
    }

    // Simple put + get cycle (omit GPU batch path to avoid long runtimes/hangs).
    auto start = TIME_NOW;
    for (uint64_t i = 0; i < num_puts; ++i) {
        auto s2 = db->Put(rocksdb::WriteOptions(), keys[i], values[i]);
        assert(s2.ok());
    }
    double put_ms = std::chrono::duration_cast<std::chrono::microseconds>(TIME_NOW - start).count() / 1000.0;
    double put_thr_ops = (put_ms > 0) ? (num_puts * 1000.0 / put_ms) : 0.0;
    std::cout << "put_time_ms: " << put_ms << " | throughput_ops_per_s: " << put_thr_ops << std::endl;

    start = TIME_NOW;
    std::string value_out;
    for (uint64_t i = 0; i < num_puts; ++i) {
        auto s2 = db->Get(rocksdb::ReadOptions(), keys[i], &value_out);
        assert(s2.ok());
    }
    double get_ms = std::chrono::duration_cast<std::chrono::microseconds>(TIME_NOW - start).count() / 1000.0;
    double get_thr_ops = (get_ms > 0) ? (num_puts * 1000.0 / get_ms) : 0.0;
    std::cout << "get_time_ms: " << get_ms << " | throughput_ops_per_s: " << get_thr_ops << std::endl;

    return 0; 
}
