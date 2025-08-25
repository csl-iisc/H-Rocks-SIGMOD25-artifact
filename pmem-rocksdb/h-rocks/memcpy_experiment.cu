#include <chrono>
#include "libgpm.cuh"
#include "gpm-helper.cuh"
#include "string.h"
#include "block_cache.cuh"
#include <iostream> 
#include <cstdio>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <functional>
#include <stdlib.h>
#include <random>
#include <getopt.h>
#include <unistd.h>
#include <algorithm>  //for std::generate_n
#include <set>


#define TIME_NOW std::chrono::high_resolution_clock::now()
using namespace std; 

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

int main(int argc, char **argv) 
{
    int option_char;
    uint64_t num_puts, NTHREADS;
    size_t value_size;
    while ((option_char = getopt (argc, argv, ":n:v:t:")) != -1) {
        switch (option_char)
        {
            case 'n': num_puts = atoi (optarg); break;
            case 'v': value_size = atoi (optarg); break;
            case 't': NTHREADS = atoi (optarg); break;
            case ':': fprintf (stderr, "option needs a value\n");
            case '?': fprintf (stderr, "usage: %s [-n <number of keys>] [-k <key size>] [-v     <value size>]\n", argv[0]);
        }
    }
     
    std::vector<char> values; 
    const auto ch_set = charset();
    std::default_random_engine rng(std::random_device{}());
    std::uniform_int_distribution<> dist(0, ch_set.size()-1);
    auto randchar = [ch_set, &dist, &rng](){return ch_set[dist(rng)];};

    std::string value = generate_random_string(value_size - 1, randchar);
    for (size_t i = 0; i < num_puts; ++i) {
        values.insert(values.end(), value.begin(), value.end());
    }

    char* str_values = values.data();
   
    const char* value_path = "values.dat"; 
    size_t valueSize = num_puts * value_size; 
    char* pm_values = (char*) gpm_map_file(value_path, valueSize, true);

    uint64_t copy_size_per_thread = valueSize/NTHREADS; 
    cout << "copy_size_per_thread: " << copy_size_per_thread << "\n"; 

    auto start = TIME_NOW; 
#pragma omp parallel for num_threads(NTHREADS)
    for(int i = 0; i < NTHREADS; ++i) 
    {
        memcpy(pm_values + i * copy_size_per_thread, str_values + i * copy_size_per_thread, copy_size_per_thread);
       //pmem_persist(pm_values + i * copy_size_per_thread, copy_size_per_thread); 
    }
    pmem_persist(pm_values, value_size); 
    auto valueCopyTime = (TIME_NOW - start).count(); 
    cout << "value_copy_time: " << valueCopyTime/1000000.0 << "\n"; 
    cudaHostRegister(str_values, valueSize, 0);

    return 0; 
}
