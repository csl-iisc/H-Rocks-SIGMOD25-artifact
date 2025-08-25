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

using namespace std;

void different_keys(viper::Viper<std::string, std::string>::Client& v_client1, viper::Viper<std::string, std::string>::Client& v_client2, viper::Viper<std::string, std::string>::Client& v_client3) {
    std::string value = "value"; 
    uint64_t num_puts = 10;
    for (uint64_t i = 0; i < num_puts - 1; ++i) {
    for(uint64_t j = 0; j < 260; ++j) {
            std::string diff_key = "key" + to_string(j); 
            v_client1.put(diff_key, value); 
        }
        for(uint64_t j = 0; j < 50; ++j) {
            std::string diff_key = "key" + to_string(i + j); 
            v_client3.put(diff_key, value); 
        }
        for(uint64_t j = 0; j < 260; ++j) {
            std::string diff_key = "key" + to_string(j * 2); 
            v_client2.put(diff_key, value); 
        }    
    }
}

void same_keys(viper::Viper<std::string, std::string>::Client& v_client1, viper::Viper<std::string, std::string>::Client& v_client2, viper::Viper<std::string, std::string>::Client& v_client3, viper::Viper<std::string, std::string>::Client& v_client, const std::string& key, std::vector<std::string>& returned_values, std::vector<std::string> values) {
    int j = 0;
    uint64_t num_puts = 101;
    std::string value = "value"; 
    uint64_t num_gets = returned_values.size();
    for (uint64_t i = 0; i < num_puts; ++i) {
        v_client2.put(key, values[i]);
        v_client1.put(key, values[i]);
        v_client2.get(key, &returned_values[j]); j++; // Returns value
        v_client2.put(key, values[i+1]);
        v_client1.remove(key); 
        v_client2.put(key, value);
        v_client3.get(key, &returned_values[j]); j++; // Returns value_i+1
        v_client1.put(key, values[i]);
        v_client.get(key, &returned_values[j]); j++; // Returns value
        v_client3.remove(key); 
        v_client2.get(key, &returned_values[j]); j++; // Returns value
        v_client3.put(key, value);
        v_client2.remove(key); 
        v_client.get(key, &returned_values[j]); j++; 
        v_client2.put(key, values[i]);
        v_client3.remove(key); 
        v_client1.remove(key); 
 }
}

int main(int argc, char** argv) {
    const size_t initial_size = 1073741824;  // 1 GiB
    auto viper_db = viper::Viper<std::string, std::string>::create("/pmem/viper_prefill_put", initial_size);

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
    auto v_client1 = viper_db->get_client();
    auto v_client2 = viper_db->get_client();
    auto v_client3 = viper_db->get_client();
    auto v_client = viper_db->get_read_only_client();

    //std::thread diff_thread(different_keys, v_client1, v_client2, v_client3);
    //std::thread same_thread(same_keys, std::ref(v_client1), std::ref(v_client2),  std::ref(v_client3),std::ref(v_client), std::ref(key), std::ref(returned_values), std::ref(values));
    std::thread diff_thread(different_keys, std::ref(v_client1), std::ref(v_client2), std::ref(v_client3));
    std::thread same_thread(same_keys, std::ref(v_client1), std::ref(v_client2),  std::ref(v_client3),std::ref(v_client), std::ref(key), std::ref(returned_values), std::ref(values));

    diff_thread.join();
    same_thread.join();

    for (uint64_t j = 0; j < num_gets; j++) {
        cout << "j: " << returned_values[j] << "\n"; 
    }
    return 0; 
}

