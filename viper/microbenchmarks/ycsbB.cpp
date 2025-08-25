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
#include <omp.h>

// Defining macros for timing analysis
#define TIME_NOW std::chrono::high_resolution_clock::now()
#define time_diff(a,b) std::chrono::duration_cast<std::chrono::microseconds>(a-b).count()

#define NUM_THREADS 10 
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
    uint64_t num_ops, nthreads;
    size_t key_size, value_size;
    // std::string file_loc; 

    while ((option_char = getopt (argc, argv, ":n:k:v:t:")) != -1){
        switch (option_char)
        {
            case 'n': num_ops = atoi (optarg); break;
            case 'k': key_size = atoi (optarg); break;
            case 'v': value_size = atoi (optarg); break;
            case 't': nthreads = atoi (optarg); break;
                      // case 'f': file_loc = optarg; break;
            case ':': fprintf (stderr, "option needs a value\n");
            case '?': fprintf (stderr, "usage: %s [-n <number of keys>] [-k <key size>] [-v <value size>]\n", argv[0]);
        }
    }

    uint64_t num_puts = num_ops * 0.1; 
    uint64_t num_gets = num_ops * 0.9; 
    num_puts = num_puts * 5;
    num_gets = num_gets * 5;
    const auto ch_set = charset();
    std::default_random_engine rng(std::random_device{}());
    std::uniform_int_distribution<> dist(0, ch_set.size()-1);
    auto randchar = [ch_set, &dist, &rng](){return ch_set[ dist(rng)];};

    std::cout << "********************* PREFILL ***************************" << std::endl;

    const size_t initial_size = 1073741824;  // 1 GiB
    auto viper_db = viper::Viper<std::string, std::string>::create("/pmem/viper", initial_size);

    // To modify records in Viper, you need to use a Viper Client.

    uint64_t num_puts_per_thread = num_puts/nthreads; 
    uint64_t num_gets_per_thread = num_gets/nthreads; 
    std::vector<std::string> keys(num_puts); 
    std::vector<std::string> values(num_puts); 

    for(uint64_t i = 0; i < num_puts; ++i) {
        keys[i] = generate_random_string(key_size, randchar);
        values[i] = generate_random_string(value_size, randchar);
    }

    
    uint64_t num_prefill = 100000000; 
    uint64_t num_prefill_per_thread = num_prefill/nthreads;
    auto start = TIME_NOW; 
#pragma omp parallel for num_threads(128)
    for(uint64_t j = 0; j < nthreads; ++j) {
    auto v_client = viper_db->get_client();
        for (uint64_t i = 0; i < num_prefill_per_thread; ++i) {
            if(j * nthreads + i >= num_puts)
                break; 
            v_client.put(keys[j * nthreads + i], values[j * nthreads + i]);
        }
    }
    auto prefill_time = (TIME_NOW - start).count(); 

    start = TIME_NOW; 
#pragma omp parallel for num_threads(NUM_THREADS)
    for(uint64_t j = 0; j < nthreads; ++j) {
    auto v_client = viper_db->get_client();
        for (uint64_t i = 0; i < num_puts_per_thread; ++i) {
            if(j * nthreads + i >= num_puts)
                break; 
            v_client.put(keys[j * nthreads + i], values[j * nthreads + i]);
            if (j * nthreads + i == num_puts/5)
                cout << "Num keys: " << j * nthreads + i << " time taken: " << (TIME_NOW - start).count()/1000000.0 << "\n";
        }
    }
    auto put_time = (TIME_NOW - start).count(); 

    cout << "PREFILL DONE\n"; 

    //std::vector<std::string> value(num_gets);
    std::string value; 
    //bool found[num_gets]; 
    bool found; 
    auto v_client = viper_db->get_client();
    start = TIME_NOW; 
#pragma omp parallel for num_threads(NUM_THREADS)
    for(uint64_t j = 0; j < nthreads; ++j) {
        for (uint64_t i = 0; i < num_gets_per_thread; ++i) {
            if(j * nthreads + i >= num_gets)
                break; 
            uint64_t idx = rand() % num_puts; 
            //cout << "i: " << i << "\t"; 
            //found[i] = v_client.get(keys[idx], &value[j * nthreads + i]);
            found = v_client.get(keys[idx], &value);
            if (j * nthreads + i == num_gets/5)
                cout << "Num keys: " << j * nthreads + i << " time taken: " << (TIME_NOW - start).count()/1000000.0 << "\n";
        }
    }
    auto get_time = (TIME_NOW - start).count(); 
    cout << "READING DONE!\n"; 
    cout << value << "\n"; 
    /*
    for(uint64_t i = 0; i < num_gets; i+=100) {
        cout << value[i] << "\n"; 
        break; 
    }
    */
    cout << "put_time: " << put_time/1000000.0 << "\n";   
    cout << "get_time: " << get_time/1000000.0 << "\n";   

    return 0; 
}
