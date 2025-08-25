#include <iostream>

#include "viper/viper.hpp"
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
#include <bits/stdc++.h>


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
    std::generate_n(str.begin(), length, rand_char);
    return str;
}

int main(int argc, char** argv) {
    int option_char;
    uint64_t num_keys;
    int nthreads; 
    // std::string file_loc; 
    uint64_t num_updates; 

    while ((option_char = getopt (argc, argv, ":n:t:u:")) != -1){
        switch (option_char)
        {
            // case 'p': p = atoi (optarg); break;
            case 'n': num_keys = atoi (optarg); break;
            case 't': nthreads = atoi (optarg); break;
            case 'u': num_updates = atoi (optarg); break;
                      // case 'f': file_loc = optarg; break;
            case ':': fprintf (stderr, "option needs a value\n");
            case '?': fprintf (stderr, "usage: %s [-n <number of keys>] [-k <key size>] [-v <value size>]\n", argv[0]);
        }
    }

    const auto ch_set = charset();
    std::default_random_engine rng(std::random_device{}());
    std::uniform_int_distribution<> dist(0, ch_set.size()-1);
    auto randchar = [ch_set, &dist, &rng](){return ch_set[ dist(rng)];};

    std::cout << "********************* PREFILL ***************************" << std::endl;

    const size_t initial_size = 1073741824;  // 1 GiB
    auto viper_db = viper::Viper<uint64_t, uint64_t>::create("/pmem/viper_prefill_put", initial_size);

    // To modify records in Viper, you need to use a Viper Client.
    uint64_t num_keys_per_thread = num_keys/nthreads; 

    std::vector<uint64_t> keys(num_keys); 
    std::vector<uint64_t> values(num_keys); 
    for(uint64_t i = 0; i < num_keys; ++i) {
        values[i] == rand()%10000000; 
        keys[i] == rand()%10000000; 
    }

    cout << "Generated kv pairs\n"; 

    auto start = TIME_NOW; 
#pragma omp parallel for num_threads(nthreads)
    for(uint64_t j = 0; j < nthreads; ++j) {
        auto v_client = viper_db->get_client();
        for (uint64_t i = 0; i < num_keys_per_thread; ++i) {
            if(j * nthreads + i > num_keys) 
                break; 
            v_client.put(keys[j * nthreads + i], values[j * nthreads + i]);
        }
    }
    auto prefill_time = (TIME_NOW - start).count(); 

    cout << "prefill_time: " << prefill_time/1000000.0 << "\n"; 
    cout << "PREFILL DONE\n"; 

    auto update_fn = [](uint64_t* value) {
        *value++; 
    }; 

    uint64_t num_updates_per_thread = num_updates/nthreads; 

    start = TIME_NOW; 
#pragma omp parallel for num_threads(nthreads)
    for(uint64_t i = 0; i < nthreads; ++i) {
        auto v_client = viper_db->get_client();
        for(uint64_t j = 0; j < num_updates_per_thread; ++j) {
            uint64_t idx = rand() % num_keys; 
            v_client.update(keys[idx], update_fn); 
        }
    }
    auto update_time = (TIME_NOW - start).count(); 

    cout << "update_time: " << update_time/1000000.0 << "\n"; 
    
    return 0; 
}
