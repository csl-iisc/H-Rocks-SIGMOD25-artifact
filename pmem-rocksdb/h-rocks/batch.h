#pragma once

#include <iostream>
#include <rocksdb/db.h>
#include <string>
#include <vector>
#include "command.h"
#include "memtable.cuh"
#include "block_cache.cuh"
#include "rocksdb/db.h"
#include <chrono>

#define TOMBSTONE_MARKER NULL
//#define BATCH_SIZE 200000000

//using namespace ROCKSDB_NAMESPACE;
class Batch{
    public: 
        Batch(std::vector<Command>& readCommands, std::vector<Command>& writeCommands, std::vector<Command>& updateCommands, uint64_t operationID, rocksdb::DB *db);
        Batch(std::vector<Command>& readCommands, std::vector<Command>& writeCommands, std::vector<Command>& updateCommands, uint64_t operationID, rocksdb::DB *db, uint64_t batchSize);

        void Get(const std::string& key);
        void Put(const std::string& key, const std::string& value);
        void Delete(const std::string& key);
        void Range(const std::string& startKey, const std::string& endKey, int num_elems);
        void Merge(const std::string& key);    
        void Exit(); 

    private: 
        void batchReads(); 
        void batchWrites(); 
        void batchWritesValues(); 
        void batchReadsValues(); 
        void batchUpdates(); 
        void batchDeletes(); 
        void batchRange(); 
        void batchAll(); 
        void initAll(); 
        void initWrites();
        void initReads();
        void initUpdates();
        void initDeletes();
        uint64_t BATCH_SIZE;

        void Execute();
        std::chrono::time_point<std::chrono::system_clock> start; 
        int timeOut; 

        std::vector<Command>& readCommands;
        std::vector<Command>& writeCommands;
        std::vector<Command>& updateCommands;
        
        BlockCache* bCache;
        BCache *cache;
        rocksdb::DB *db; 

        void processPuts(); 
        void processGets(); 
        void processRange(); 
        void processDeletes(); 
        void processUpdates(); 
        void ifBatchSizeReached(); 
        void reset(); 

        uint64_t operationID = 0;
        int keyLength = 0; 
        int valueLength = 0; 

        int readKeySize = 0;
        int writeKeySize = 0;
        int updateKeySize = 0;

        int readValueSize = 0;
        int writeValueSize = 0;
        int updateValueSize = 0;

        uint64_t numWrites = 0; 
        uint64_t numReads = 0; 
        uint64_t numUpdates = 0;

        int batchSize = 0;
        int batchID = 0; 
        bool putsWithValues = false;
        
        std::vector<char> tPutKeys; 
        std::vector<char> tPutValues; 

        uint64_t* putOperationIds = nullptr;
        uint64_t* getOperationIds = nullptr;
        uint64_t* deleteOperationIds = nullptr;
        uint64_t* rangeOperationIds = nullptr;
        uint64_t* updateOperationIds = nullptr;

        PutCommand putCommand; 
        GetCommand getCommand; 
        DeleteCommand deleteCommand; 
        MergeCommand mergeCommand; 
        RangeCommand rangeCommand; 
 
        char** putValuePointers = nullptr;

        Memtable *immutableMemtable;
        Memtable *activeMemtable;
        Memtable *snapshot;
        
        UpdateMemtable *updateMemtable;
}; 
