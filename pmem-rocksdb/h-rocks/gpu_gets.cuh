#pragma once 
#include "gpu_gets.cuh"
#include "memtable.cuh"
#include "block_cache.cuh"
//void searchGetsOnGPU(char* getKeys, Memtable activeMemtable, Memtable immutableMemtable, int keyLength, int numReads, char** getValuePointers); 
void searchGetsOnGPU(char* getKeys, Memtable *activeMemtable, Memtable *immutableMemtable, int keyLength, uint64_t numReads, char** getValuePointers, uint64_t* getOperationIds, BCache* cache); 
//void searchGetsOnGPU(GetCommand getCommand, Memtable &activeMemtable, Memtable &immutableMemtable,BCache* cache); 
