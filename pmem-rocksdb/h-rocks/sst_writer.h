#pragma once 
#include <iostream>
#include "rocksdb/db.h"
#include "block_cache.cuh"


//int sstWriter(char *keys, char **values, uint64_t nkeys, uint32_t keyLen, uint32_t valueLen, int NTHREADS); 
int sstWriter(char *keys, char **values, uint64_t nkeys, uint32_t keyLen, uint32_t valueLen, int NTHREADS, rocksdb::DB *db, BlockCache *bCache, BCache *cache); 
