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


int main(int argc, char** argv) {

    const size_t initial_size = 1073741824;  // 1 GiB
    auto viper_db = viper::Viper<uint64_t, uint64_t>::create("/pmem/viper_prefill_put", initial_size);

    // To modify records in Viper, you need to use a Viper Client.
    std::cout << "Generated kv pairs\n"; 
    uint64_t key = 0;
    uint64_t value = 0; 
    auto v_client = viper_db->get_client();
    v_client.put(key, value);

    auto update_fn = [](uint64_t* value) {
        *value++; 
    }; 
    int num_updates = 100; 

    uint64_t returned_value; 
#pragma omp parallel for num_threads(2)
    for(uint64_t j = 0; j < num_updates; ++j) {
            //v_client.update(key, update_fn); 
            uint64_t returned_value; 
            v_client.get(key, &returned_value); 
            returned_value++; 
            v_client.put(key, returned_value); 
        }
    v_client.get(key, &returned_value); 
    std::cout << returned_value << "\n"; 

    return 0; 
}
