#include <iostream>
#include <chrono>
#include <mutex>
#include <fstream>
#include <filesystem>
#include <unordered_map>
#include <string>
#include "../src/hashtable/Hashtable.h"

#include <cstdio>
#include <fstream>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <functional>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <getopt.h>
#include <unistd.h>


using namespace std;
#define TIME_NOW std::chrono::high_resolution_clock::now()
#define time_diff(a,b) std::chrono::duration_cast<std::chrono::microseconds>(a-b).count()


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
    uint64_t num_puts, num_updates;
    int nthreads; 
    int batch;
    int ratio;

    // Declare R and provide seed

    while ((option_char = getopt (argc, argv, ":p:u:t:")) != -1){
        switch (option_char)
        {
            case 'p': num_puts = atoi (optarg); break;
            case 'u': num_updates = atoi (optarg); break;
            case 't': nthreads = atoi (optarg); break;
            case ':': fprintf (stderr, "option needs a value\n");
            case '?': fprintf (stderr, "usage: %s [-n <number of keys>] [-k <key size>] [-v <value size>] [-b <batch size>] [-r <put to get ratio>]\n", argv[0]);
        }
    }

    std::cout << "Number of keys: " << num_puts << std::endl;

    // A hash table with 8 byte keys and values
    const std::string data_location = "/pmem/plush_table";
    Hashtable<uint64_t, uint64_t, PartitionType::Hash> table("/pmem/plush_table", true); 

    std::vector<uint64_t> keys(num_puts);
    std::vector<uint64_t> values(num_puts);

    uint64_t num_puts_per_thread = num_puts/nthreads; 

#pragma omp parallel for num_threads(nthreads)
    for(uint64_t j = 0; j < nthreads; ++j) {
        for(uint64_t i = 0; i < num_puts_per_thread; ++i) {
            if(j * nthreads + i >= num_puts)
                break; 
            keys[j * nthreads + i] = rand() % 10000000;
            values[j * nthreads + i] = rand() % 10000000;
            table.insert(keys[j * nthreads + i], values[j * nthreads + i]);
        }
    }

    cout << "************ PREFILL DONE ************\n"; 
    uint64_t num_updates_per_thread = num_updates/nthreads; 
    uint64_t value;

    auto start = TIME_NOW; 
#pragma omp parallel for num_threads(nthreads)
    for(uint64_t j = 0; j < nthreads; ++j) {
        for(uint64_t i = 0; i < num_updates_per_thread; ++i) {
            if(j * nthreads + i >= num_updates)
                break; 
            uint64_t idx = rand() % num_puts; 
            table.lookup(keys[idx], reinterpret_cast<uint8_t *>(&value)); 
            value++; 
            table.insert(keys[idx], value); 
        }
    }
    auto update_time = (TIME_NOW - start).count(); 

    cout << "update_time: " << update_time/1000000.0 << "\n"; 

    return 0; 
} 
