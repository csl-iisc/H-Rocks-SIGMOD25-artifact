#pragma once 
#include "memtable.cuh"
#include "block_cache.cuh"
void get_values(char* getKeys, MemtableWithValues *activeMemtable, MemtableWithValues *immutableMemtable, int keyLength, uint64_t numReads, uint64_t* getOperationIds, BCache* cache); 
