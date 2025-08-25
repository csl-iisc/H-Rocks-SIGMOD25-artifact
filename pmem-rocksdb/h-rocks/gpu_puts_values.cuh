#pragma once 
#include "command.h"
#include "memtable.cuh"

void sortPutsOnGPUWithValues(PutCommand &putCommand, uint64_t num_elems, uint64_t keyLength, MemtableWithValues *table, uint64_t batchID, int valueLength);
