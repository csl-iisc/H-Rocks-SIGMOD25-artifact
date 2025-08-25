#include <rocksdb/db.h>
#include <rocksdb/slice.h>
#include <string.h>
#include <iostream>
#include <unistd.h>
#include <bits/stdc++.h>
#include "block_cache.cuh"

#define NOT_FOUND NULL 

// TODO: update this
// Pass cache variable 
void cpu_get(char* keys, char** values, uint32_t keyLen, uint32_t valueLen, unsigned int num_key_not_found, uint64_t* key_not_found, rocksdb::DB *db, BCache *cache) 
{
    std::cout << "num keys not found: " << num_key_not_found << "\n"; 
    if(num_key_not_found == 0) 
        return; 
#pragma omp parallel for num_threads(128)
    for(uint64_t i = 0; i < num_key_not_found; ++i) 
    {
        std::string str_key, value; 
        str_key.assign(keys + key_not_found[i] * keyLen); 
        rocksdb::Slice key = str_key; 
        // Get value from DB 
        rocksdb::Status s = db->Get(rocksdb::ReadOptions(), key, &value); 
        // Update value 
        strcpy(values[i], value.c_str()); 
        // Add key and value to the block cache. 
        cache->put(cache, str_key, value.c_str()); 
    }
}

