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
    uint64_t num_puts, num_gets;
    int nthreads; 
    int batch;
    int ratio;
    size_t key_size, value_size;

    // Declare R and provide seed

    while ((option_char = getopt (argc, argv, ":n:k:v:p:t:")) != -1){
        switch (option_char)
        {
            case 'p': num_puts = atoi (optarg); break;
            case 'n': num_gets = atoi (optarg); break;
            case 'k': key_size = atoi (optarg); break;
            case 'v': value_size = atoi (optarg); break;
            case 't': nthreads = atoi (optarg); break;
            case ':': fprintf (stderr, "option needs a value\n");
            case '?': fprintf (stderr, "usage: %s [-n <number of keys>] [-k <key size>] [-v <value size>] [-b <batch size>] [-r <put to get ratio>]\n", argv[0]);
        }
    }

    std::cout << "Number of keys: " << num_puts << std::endl;
    std::cout << "Key size: " << key_size << std::endl;
    std::cout << "Value size: "<< value_size << std::endl;

    // A hash table with 8 byte keys and values
    const std::string data_location = "/pmem/plush_table";
    Hashtable<std::span<const std::byte>, std::span<const std::byte>, PartitionType::Hash> table("/pmem/plush_table", true); 

    const auto ch_set = charset();
    std::default_random_engine rng(std::random_device{}());
    std::uniform_int_distribution<> dist(0, ch_set.size()-1);
    auto randchar = [ch_set, &dist, &rng](){return ch_set[ dist(rng) ];};

    std::vector<std::string> keys(num_puts);
    std::vector<std::string> values(num_puts);

    for(uint64_t i = 0; i < num_puts; ++i) {
        keys[i] = generate_random_string(key_size,randchar);
        values[i] = generate_random_string(value_size,randchar);
    }

    for(uint64_t i = 0; i < num_puts; ++i) {
        table.insert(std::span<std::byte>(reinterpret_cast<std::byte*>(keys[i].data()), keys[i].size()), std::span<std::byte>(reinterpret_cast<std::byte*>(values[i].data()), values[i].size()));
    }

    cout << "************ PREFILL DONE ************\n"; 

    auto start = TIME_NOW; 
    for(uint64_t i = 0; i < num_gets; ++i) {
        uint64_t idx = rand() % num_puts; 
        std::unique_ptr<uint8_t[]> search_result = std::make_unique<uint8_t[]>(1e6);
        table.lookup(std::span<std::byte>(reinterpret_cast<std::byte*>(keys[idx].data()), keys[idx].size()), search_result.get());
    }
    auto get_time = (TIME_NOW - start).count(); 

    cout << "get_time: " << get_time/1000000.0 << "\n"; 

    return 0; 
} 
