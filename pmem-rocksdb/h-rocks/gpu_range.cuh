#pragma once 
#include "memtable.cuh"

void rangeOnGPU(char* startKeys, char* endKeys, Memtable *activeMemtable, Memtable *immutableMemtable, int keyLength, uint64_t numRange, char** &rangeValuePointers, uint64_t* rangeOperationIds); 
