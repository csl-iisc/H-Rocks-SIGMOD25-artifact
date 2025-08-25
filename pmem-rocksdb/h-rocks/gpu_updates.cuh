#pragma once 
#include "command.h"
#include "memtable.cuh"
#include "batch.h"

void gpuIncrements(Memtable* activeMemtable, Memtable* immutableMemtable, uint64_t numUpdates, MergeCommand& mergeCommand, int batchID, BCache* cache); 
