#include <iostream>
#include <chrono>
#include <thread>
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

#include "batch.h"
#include <rocksdb/db.h>
#include <rocksdb/slice.h>
#include <rocksdb/options.h>

// Defining macros for timing analysis
#define TIME_NOW std::chrono::high_resolution_clock::now()
#define time_diff(a,b) std::chrono::duration_cast<std::chrono::microseconds>(a-b).count()

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


int main(int argc, char **argv) {
    int option_char;
    uint64_t n, p;
    size_t key_size, value_size;
    std::string file_loc; 

    while ((option_char = getopt (argc, argv, ":p:n:k:v:f:")) != -1){
        switch (option_char)
        {
            case 'p': p = atoi (optarg); break;
            case 'n': n = atoi (optarg); break;
            case 'k': key_size = atoi (optarg); break;
            case 'v': value_size = atoi (optarg); break;
            case 'f': file_loc = optarg; break;
            case ':': fprintf (stderr, "option needs a value\n");
            case '?': fprintf (stderr, "usage: %s [-n <number of keys>] [-k <key size>] [-v <value size>]\n", argv[0]);
        }
    }
    std::cout << "Number of keys: " << n << std::endl;
    std::cout << "Key size: " << key_size << std::endl;
    std::cout << "Value size: " << value_size << std::endl;

    // Declaring rocksdb
    rocksdb::DB* db;
    rocksdb::Options options;
    // Optimize RocksDB. This is the easiest way to get RocksDB to perform well
    options.IncreaseParallelism();
    options.OptimizeLevelStyleCompaction();
    // Create the DB if it's not already present
    options.create_if_missing = true;
    // Open db
    std::string file_name = "/pmem/" + file_loc; 
    cout << file_name; 
    auto start = TIME_NOW; 
    rocksdb::Status s = rocksdb::DB::Open(options, file_name, &db);
    assert(s.ok());
    auto db_open_time = (TIME_NOW - start).count(); 
    std::cout << "db opened." << std::endl;
    std::cout << "db open time: " << db_open_time/1000000.0 << std::endl; 

    const auto ch_set = charset();
    std::default_random_engine rng(std::random_device{}());
    std::uniform_int_distribution<> dist(0, ch_set.size()-1);
    auto randchar = [ch_set, &dist, &rng](){return ch_set[ dist(rng) ];};
 
    uint64_t i = 0; 
    std::vector<std::string> keys(n); 
    std::vector<std::string> values(n); 
    for(i = 0; i < n; ++i) {
        keys[i] = generate_random_string(key_size -1, randchar);
        values[i] = generate_random_string(value_size - 1, randchar);
    }
    

    // Calculate the delay between requests to distribute them uniformly across a second
    //auto delay = std::chrono::milliseconds(1000 / n);
    auto delay = std::chrono::microseconds(1000000 / n);  // 1 second = 1 million microseconds

    std::cout << "num keys: " << n << "\n"; 

    char *key, *value; 
    key = (char*)malloc(key_size); 
    value = (char*)malloc(value_size); 
     std::vector<Command> readCommands, writeCommands, updateCommands;
    Batch batch(readCommands, writeCommands, updateCommands, 0, db);
   
    start = TIME_NOW; 
    for (int i = 0; i < n; ++i) {

        // Send PUT request
        strcpy(key, keys[i].c_str()); 
        strcpy(value, values[i].c_str()); 
     
        batch.Put(key, value); 
        // Sleep for the calculated delay, except after the last request
        if (i < n - 1) {
            std::this_thread::sleep_for(delay);
        }
    }
    batch.Exit(); 
    auto write_time = (TIME_NOW - start).count(); 
    std::cout << "write time: " << write_time/1000000.0 << " millisecond" << std::endl; 

    return 0;
}

