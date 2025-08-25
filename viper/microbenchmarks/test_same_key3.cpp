#include <iostream>
#include <thread>
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
#include <typeinfo>


using namespace std;
const size_t initial_size = 1073741824;  // 1 GiB
auto viper_db = viper::Viper<std::string, std::string>::create("/pmem/viper_prefill_put", initial_size);
auto v_client1 = viper_db->get_client();
auto v_client = viper_db->get_client();

void different_keys() {
    std::string value1 = "valuhiQLEHNlqehLQXqlxQNLHXlqehxLEHlqhinEEedgjjWDGQLEYIQNELXQEIHILEKnldhlkaehlQHEIqlhneqJELqe"; 
    uint64_t num_puts = 10;
    for (uint64_t i = 0; i < num_puts - 1; ++i) {
        for(uint64_t j = 0; j < 260; ++j) {
            std::string diff_key = "key" + to_string(j); 
            //v_client1.put("same_key", "value"); 
            v_client1.put(diff_key, value1); 
            v_client1.put(diff_key, value1); 
        }
        for(uint64_t j = 0; j < 500; ++j) {
            std::string diff_key = "key" + to_string(i + j); 
            v_client1.put(diff_key, value1); 
        }
        for(uint64_t j = 0; j < 260; ++j) {
            std::string diff_key = "key" + to_string(j * 2); 
            v_client1.put(diff_key, value1); 
        }    
    }
    cout << "Reached diff keys\n"; 
}

void same_keys(std::vector<std::string> returned_values, std::vector<std::string> values, uint64_t num_puts) {
    auto v_client2 = viper_db->get_client();
    auto v_client3 = viper_db->get_client();
    uint64_t num_gets = num_puts * 5;
    int k = 0;
    std::string value = "value"; 
    std::string key = "same_key"; 
    for (uint64_t i = 0; i < num_puts - 1; ++i) {
        v_client.put(key, values[i]);
        v_client.get(key, &returned_values[k]); k++; // Returns value0
        v_client.put(key, values[i]);
        v_client1.remove(key); 
        v_client.get(key, &returned_values[k]); k++; // Returns EMPTY
        v_client.remove(key); 
        v_client.put(key, value);
        v_client2.remove(key); 
        v_client3.put(key, values[i+1]);
        v_client3.get(key, &returned_values[k]); k++; // Returns value1
        v_client.put(key, values[i]);
        v_client2.put(key, value);
        v_client.get(key, &returned_values[k]); k++; // Returns value
        v_client.remove(key); 
        v_client3.remove(key); 
        v_client2.put(key, value);
        v_client.put(key, values[i]);
        v_client.remove(key);
        v_client.get(key, &returned_values[k]); k++; // Returns EMPTY
    }
    for (uint64_t l = 0; l < num_gets; l++) {
        cout << "j: " << returned_values[l] << "\n"; 
    }
    cout << "Reached here\n"; 
}

int main(int argc, char** argv) {
    uint64_t num_ops = 101;
    uint64_t num_puts = num_ops / 2;
    uint64_t num_gets = num_puts * 5;
    std::string key = "same_key";
    std::vector<std::string> values(num_puts);
    std::vector<std::string> returned_values(num_gets);
    for (uint64_t i = 0; i < num_puts; ++i) {
        values[i] = "value" + to_string(i);
    }

    cout << "Generated kv pairs\n";
     std::thread same_thread(same_keys, returned_values, values, num_puts);   
     std::thread diff_thread(different_keys);

    same_thread.join();
    diff_thread.join();
    return 0; 
}
