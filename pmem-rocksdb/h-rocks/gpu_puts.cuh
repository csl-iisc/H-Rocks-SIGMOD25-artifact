#pragma once
#include "batch.h"
#include "memtable.cuh"
#include "block_cache.cuh"

//void sortPutsOnGPU(char *putKeys, char **putValuePointers, uint64_t *putOperationIDs, int size, int keyLength, Memtable& activeMemtable, int batchID); 

void sortPutsOnGPU(PutCommand &putCommand, uint64_t num_elems, uint64_t keyLength, Memtable *table, uint64_t batchID, BCache *cache);
