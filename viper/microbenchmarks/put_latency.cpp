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
    size_t key_size, value_size;
    // std::string file_loc; 

    while ((option_char = getopt (argc, argv, ":n:k:v:t:")) != -1){
        switch (option_char)
        {
            // case 'p': p = atoi (optarg); break;
            case 'n': num_keys = atoi (optarg); break;
            case 'k': key_size = atoi (optarg); break;
            case 'v': value_size = atoi (optarg); break;
            case 't': nthreads = atoi (optarg); break;
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

    // To modify records in Viper, you need to use a Viper Client.
    uint64_t num_keys_per_thread = num_keys/nthreads; 

    std::vector<std::string> keys(num_keys); 
    std::vector<std::string> values(num_keys); 
    for(uint64_t i = 0; i < num_keys; ++i) {
        keys[i] = generate_random_string(key_size, randchar);
        values[i] = generate_random_string(value_size, randchar);
    }

    cout << "Generated kv pairs\n"; 
    std::vector<double> latencies;

    // ... existing code ...
    auto viper_db = viper::Viper<std::string, std::string>::create("/pmem/viper_prefill_put", initial_size);

    auto v_client = viper_db->get_client();
    auto put_start = TIME_NOW;

    double time_per_key = 1.0 / num_keys;

    for (uint64_t i = 0; i < num_keys; ++i) {
        v_client.put(keys[i], values[i]);
        auto put_end = TIME_NOW;
        double latency = time_diff(put_end, put_start + std::chrono::duration<double>(time_per_key * i));
        latencies.push_back(latency);

        // Sleep for the remaining time per key
        std::this_thread::sleep_for(std::chrono::duration<double>(time_per_key - latency));
    }
    

    // ... existing code ...

    // Calculate p50 and p95 latencies
    std::sort(latencies.begin(), latencies.end());
    size_t p50_index = latencies.size() * 0.5;
    size_t p95_index = latencies.size() * 0.95;
    size_t p90_index = latencies.size() * 0.90;
    double p50_latency = latencies[p50_index];
    double p95_latency = latencies[p95_index];
    double p90_latency = latencies[p90_index];

    std::cout << "p50_latency: " << p50_latency/1000000.0 << std::endl;
    std::cout << "p90_latency: " << p90_latency/1000000.0 << std::endl;
    std::cout << "p95_latency: " << p95_latency/1000000.0 << std::endl;

    return 0; 
}
