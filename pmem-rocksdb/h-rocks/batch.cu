#include "batch.h"
#include "command.h"
#include "gpu_puts.cuh"
#include "gpu_gets.cuh"
#include "gpu_updates.cuh"
#include "block_cache.cuh"
#include "sst_writer.h"
#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "rocksdb/sst_file_writer.h"
#include "gpu_range.cuh"
#include "gpu_puts_values.cuh"
#include "gpu_gets_values.cuh"
#include "libgpm.cuh"
#include <cstring>
#include <strings.h>

#define TIME_NOW std::chrono::high_resolution_clock::now()
#define NTHREADS 100

using namespace ROCKSDB_NAMESPACE; 

Batch::Batch(std::vector<Command>& readCommands, std::vector<Command>& writeCommands, std::vector<Command>& updateCommands, uint64_t operationID, rocksdb::DB *db_)
    : readCommands(readCommands), writeCommands(writeCommands), updateCommands(updateCommands), operationID(0), db(db_) {
        start = TIME_NOW;
        timeOut = 100000000;
        //BlockCache bCache; 
        cache = bCache->createCache(cache); 
        db = db_; 
        BATCH_SIZE = 200000000; 
        numReads = 0; 
        numWrites = 0; 
        writeCommands.reserve(BATCH_SIZE); 
        readCommands.reserve(BATCH_SIZE); 
        updateCommands.reserve(BATCH_SIZE); 
        putCommand.tKeys.reserve(BATCH_SIZE * 8); 
        putCommand.tValues.reserve(BATCH_SIZE * 8); 
        putCommand.tOpID.reserve(BATCH_SIZE); 

        cudaMallocManaged(&activeMemtable, sizeof(Memtable)); 
        activeMemtable->batchID = -1; 
        cudaMallocManaged(&immutableMemtable, sizeof(Memtable)); 
        immutableMemtable->batchID = -1; 
    }

Batch::Batch(std::vector<Command>& readCommands, std::vector<Command>& writeCommands, std::vector<Command>& updateCommands, uint64_t operationID, rocksdb::DB *db_, uint64_t batchSize)
    : readCommands(readCommands), writeCommands(writeCommands), updateCommands(updateCommands), operationID(0), db(db_) {
        start = TIME_NOW;
        timeOut = 1000000;
        BlockCache bCache; 
        cache = bCache.createCache(cache); 
        db = db_; 
        BATCH_SIZE = batchSize;
        numReads = 0; 
        numWrites = 0; 
        writeCommands.reserve(BATCH_SIZE); 
        readCommands.reserve(BATCH_SIZE); 
        updateCommands.reserve(BATCH_SIZE); 
        putCommand.tKeys.reserve(BATCH_SIZE * 8); 
        putCommand.tValues.reserve(BATCH_SIZE * 8); 
        putCommand.tOpID.reserve(BATCH_SIZE); 
         cudaMallocManaged(&activeMemtable, sizeof(Memtable)); 
        activeMemtable->batchID = -1; 
        cudaMallocManaged(&immutableMemtable, sizeof(Memtable)); 
        immutableMemtable->batchID = -1; 
    }


void Batch::Put(const std::string& key, const std::string& value) {
    Command command(Type::PUT, key, value, operationID++);
    writeCommands.push_back(command);
    writeKeySize += command.key.size() + 1;
    writeValueSize += command.value.size() + 1;
    keyLength = command.key.length() + 1;
    valueLength = command.value.length() + 1;
    putCommand.tKeys.insert(putCommand.tKeys.end(), key.begin(), key.end());
    putCommand.tKeys.push_back('\0'); 
    putCommand.tValues.insert(putCommand.tValues.end(), value.begin(), value.end());
    putCommand.tValues.push_back('\0'); 
    putCommand.tOpID.push_back(operationID); 
    putCommand.numPuts++; 
    numWrites++; 
    //std::cout << "putting operationID: " << operationID << " key: " << key << "\n"; 
    ifBatchSizeReached(); 
}

void Batch::Get(const std::string& key) {
    Command command(Type::GET, key, operationID++);
    readCommands.push_back(command);
    readKeySize += command.key.size() + 1;
    keyLength = command.key.length() + 1;
    getCommand.tKeys.insert(getCommand.tKeys.end(), key.begin(), key.end());
    getCommand.tKeys.push_back('\0'); 
    getCommand.tOpID.push_back(operationID); 
    getCommand.numGets++; 
    //std::cout << "getting operationID: " << operationID << " key: " << key << "\n"; 
    numReads++; 
    ifBatchSizeReached(); 
}

void Batch::Delete(const std::string& key) {
    Command command(Type::DELETE, key, operationID++);
    writeCommands.push_back(command);
    writeKeySize += command.key.size() + 1;
    deleteCommand.numDeletes++; 
    deleteCommand.tKeys.insert(deleteCommand.tKeys.end(), key.begin(), key.end());
    deleteCommand.tKeys.push_back('\0'); 
    numWrites++; 
    ifBatchSizeReached(); 
}

void Batch::Range(const std::string& startKey, const std::string& endKey, int num_elems) {
    Command command(Type::RANGE, startKey, endKey, num_elems, operationID++);
    readCommands.push_back(command);
    readKeySize += command.key.size() + 1;
    rangeCommand.tStartKeys.insert(rangeCommand.tStartKeys.end(), startKey.begin(), startKey.end());
    rangeCommand.tStartKeys.push_back('\0'); 
    rangeCommand.tEndKeys.insert(rangeCommand.tEndKeys.end(), endKey.begin(), endKey.end());
    rangeCommand.tEndKeys.push_back('\0'); 
    rangeCommand.tOpID.push_back(operationID++); 
    rangeCommand.numRangeQs++; 
    ifBatchSizeReached(); 
}

void Batch::Merge(const std::string& key) {
    Command command(Type::UPDATE, key, operationID++);
    updateCommands.push_back(command);
    updateKeySize += command.key.size() + 1;
    mergeCommand.tKeys.insert(mergeCommand.tKeys.end(), key.begin(), key.end());
    mergeCommand.tKeys.push_back('\0'); 
    mergeCommand.tOpID.push_back(operationID++); 
    mergeCommand.numMerges++; 
    numUpdates++; 
    ifBatchSizeReached(); 
}

void Batch::ifBatchSizeReached() 
{
    auto elapsed_time = (TIME_NOW - start).count();
    if(numReads + numWrites + numUpdates >= BATCH_SIZE || elapsed_time/1000000.0 > timeOut) {
        std::cout << "elapsed time: " << elapsed_time/1000000.0 << "\n"; 
        std::cout << "batch size reached\n"; 
        batchID++; 
        std::cout << "batchID: " << batchID << "\n"; 
        Execute(); 
    }
}

void Batch::initWrites() 
{
    putCommand.valuePtrs = (char**) malloc(numWrites * sizeof(char*)); 
}

void Batch::initDeletes()
{
    deleteCommand.valuePtrs = (char**) malloc(deleteCommand.numDeletes * sizeof(char*)); 
}

void Batch::initReads()
{
}

void Batch::initUpdates()
{
}

void Batch::batchWritesValues()
{
    // No need to setup value pointers
    auto start = TIME_NOW; 
    putCommand.keys = putCommand.tKeys.data();  
    putCommand.values = putCommand.tValues.data();  
    putCommand.operationIDs = putCommand.tOpID.data(); 

    auto write_batch_time = (TIME_NOW - start).count(); 
    std::string valueFileName = "value" + std::to_string(batchID);
    char* fileName = new char[valueFileName.length() + 1]; // Allocate memory for the string
    strcpy(fileName, valueFileName.c_str());
    size_t valueSize = numWrites * valueLength; 
    char* pm_values = (char*) gpm_map_file(fileName, valueSize, true);

    size_t num_transfer_threads = 4; 
    if (valueLength > 32)
        num_transfer_threads = 8; 
    uint64_t copy_size_per_thread = valueSize/num_transfer_threads; 

    start = TIME_NOW; 
#pragma omp parallel for num_threads(num_transfer_threads)
    for(int i = 0; i < num_transfer_threads; i++) {
        memcpy(pm_values + i * copy_size_per_thread, putCommand.values + i * copy_size_per_thread, copy_size_per_thread);
    }
    auto valueCopyTime = (TIME_NOW - start).count(); 
    std::cout << "value_copy_time: " << valueCopyTime/1000000.0 << "\n"; 

    cudaHostRegister(putCommand.values, numWrites * valueLength, 0); 
    putCommand.operationIDs = putCommand.tOpID.data(); 
   std::cout << "write_batch_time: " << write_batch_time/1000000.0 << "\n";
}

void Batch::batchWrites()
{
    putCommand.keys = putCommand.tKeys.data();  
    putCommand.values = putCommand.tValues.data();  
    putCommand.valueLength = valueLength; 
    putCommand.keyLength = keyLength; 

    const char* valuePath = "values.dat"; 
    size_t valueSize = numWrites * valueLength; 
    char* pm_values = (char*) gpm_map_file(valuePath, valueSize, true);

    size_t num_transfer_threads = 8; 
    uint64_t copy_size_per_thread = valueSize/num_transfer_threads; 

    auto start = TIME_NOW; 
#pragma omp parallel for num_threads(num_transfer_threads)
    for(int i = 0; i < num_transfer_threads; i++) {
        memcpy(pm_values + i * copy_size_per_thread, putCommand.values + i * copy_size_per_thread, copy_size_per_thread);
    }
    pmem_mt_persist(pm_values, valueSize); 
    auto valueCopyTime = (TIME_NOW - start).count(); 
    std::cout << "value_copy_time: " << valueCopyTime/1000000.0 << "\n"; 

    cudaHostRegister(putCommand.values, numWrites * valueLength, 0); 
    putCommand.operationIDs = putCommand.tOpID.data(); 

    start = TIME_NOW; 
    if(numWrites < 1000000) {
        for(uint64_t i = 0; i < numWrites; ++i) {
            Command &cmd = writeCommands[i];
            if (cmd.type == Type::PUT) {
                putCommand.valuePtrs[i] = putCommand.values + i * valueLength; 
            } 
        }
    } else {
#pragma omp parallel for num_threads(120)
        for(uint64_t i = 0; i < numWrites; ++i) {
            putCommand.valuePtrs[i] = putCommand.values + i * valueLength; 
#ifdef __PRINT_DEBUG__
            std::cout << "Value: " << putCommand.valuePtrs[i] << "\n"; 
            std::string currentKey(putCommand.keys + putKeyIndex, cmd.key.length());
            std::cout << "writing: " << i << " key: " << currentKey << " put key idx: " << putKeyIndex << " opId: " << putCommand.operationIDs[i] << "\n"; 
#endif

        }
    }
    auto write_batch_time = (TIME_NOW - start).count(); 
    std::cout << "write_batch_time: " << write_batch_time/1000000.0 << "\n";
    //auto write_batch_time = (TIME_NOW - start).count(); 
    //std::cout << "write_batch_time: " << write_batch_time/1000000.0 << "\n";
}

void Batch::batchDeletes()
{
    start = TIME_NOW; 
    deleteCommand.keys = deleteCommand.tKeys.data();  
    deleteCommand.operationIDs = deleteCommand.tOpID.data(); 
    memset(deleteCommand.valuePtrs, 0, deleteCommand.numDeletes * sizeof(char*)); 
    auto delete_batch_time = (TIME_NOW - start).count(); 
    std::cout << "delete_batch_time: " << delete_batch_time/1000000.0 << "\n";
}

void Batch::batchRange()
{
    start = TIME_NOW; 
    rangeCommand.startKeys = rangeCommand.tStartKeys.data();  
    rangeCommand.endKeys = rangeCommand.tEndKeys.data();  
    rangeCommand.operationIDs = rangeCommand.tOpID.data(); 
    auto range_batch_time = (TIME_NOW - start).count(); 
    std::cout << "range_batch_time: " << range_batch_time/1000000.0 << "\n";
}


void Batch::batchReads()
{
    start = TIME_NOW; 
    //numReads = readCommands.size(); 
    getCommand.keys = getCommand.tKeys.data();  
    getCommand.operationIDs = getCommand.tOpID.data(); 
    auto read_batch_time = (TIME_NOW - start).count(); 
    std::cout << "read_batch_time: " << read_batch_time/1000000.0 << "\n";
}

void Batch::batchReadsValues()
{
    start = TIME_NOW; 
    //numReads = readCommands.size(); 
    getCommand.keys = getCommand.tKeys.data();  
    getCommand.operationIDs = getCommand.tOpID.data(); 
    auto read_batch_time = (TIME_NOW - start).count(); 
    std::cout << "read_batch_time: " << read_batch_time/1000000.0 << "\n";
}

void Batch::batchUpdates()
{
    start = TIME_NOW; 
    //numReads = readCommands.size(); 
    mergeCommand.keys = mergeCommand.tKeys.data();  
    mergeCommand.operationIDs = mergeCommand.tOpID.data(); 
    auto update_batch_time = (TIME_NOW - start).count(); 
    std::cout << "update_batch_time: " << update_batch_time/1000000.0 << "\n";
}

void Batch::Execute() 
{
    const char* use_values = std::getenv("HR_PUTS_WITH_VALUES");
    putsWithValues = (use_values &&
        (strcmp(use_values, "1") == 0 || strcasecmp(use_values, "true") == 0));
    if(putsWithValues) {
        std::cout << "Here\n"; 
        batchWritesValues(); 
        initReads();
        batchReadsValues(); 
        processPuts(); 
        //processGets(); 
        return; 
    }
    initWrites();
    batchWrites(); 
    initDeletes();
    batchDeletes(); 
    initUpdates(); 
    batchUpdates(); 
    initReads(); 
    batchReads(); 
    batchRange(); 

    processPuts(); 
    processDeletes(); 
    processUpdates(); 
    processGets(); 
    processRange(); 

    numWrites = 0; 
    numReads = 0; 
    numUpdates = 0; 


    immutableMemtable = activeMemtable; 
    // TODO: Check if the memtable is all flushed? And then delete it. 
    deleteMemtable(activeMemtable); 

    /*
       delete[] putCommand.keys;
       delete[] getCommand.keys;
       delete[] mergeCommand.keys;
     */
}

void Batch::processPuts()
{
    //putsWithValues = true; 
    if (putsWithValues) {
        std::cout << "Here\n"; 
        MemtableWithValues *active; 
        MemtableWithValues *immutable; 
        cudaMallocManaged(&active, sizeof(MemtableWithValues)); 
        cudaMallocManaged(&immutable, sizeof(MemtableWithValues)); 
        active->size = writeCommands.size();
        active->valueLength = valueLength; 
        immutable->size = 0;
        sortPutsOnGPUWithValues(putCommand, writeCommands.size(), keyLength, active, batchID, valueLength);  
        get_values(getCommand.keys, active, immutable, keyLength, numReads, getCommand.operationIDs, cache); 
        return; 
    }
    
    activeMemtable->size = writeCommands.size();
    activeMemtable->keyLength = keyLength; 
    activeMemtable->valueLength = valueLength; 
    activeMemtable->batchID = batchID; 
    //numWrites = writeCommands.size(); 
    sortPutsOnGPU(putCommand, writeCommands.size(), keyLength, activeMemtable, batchID, cache);  
    auto start = TIME_NOW;
    char *sortedKeys, **sortedValues; 
    sortedKeys = (char*)malloc(numWrites * keyLength); 
    sortedValues = (char**)malloc(numWrites * sizeof(char*)); 
    cudaHostRegister(sortedKeys, numWrites * keyLength, 0); 
    cudaHostRegister(sortedValues, numWrites * sizeof(char*), 0); 
    cudaMemcpy(sortedKeys, activeMemtable->d_sortedKeys, numWrites * keyLength, cudaMemcpyDeviceToHost); 
    cudaMemcpy(sortedValues, activeMemtable->d_sortedValuePointers, numWrites * sizeof(char*), cudaMemcpyDeviceToHost); 
    cudaError_t err = cudaPeekAtLastError();
    printf("Error %d cudaPeekError\n", err); 
    std::cout << "sorted keys: " << sortedKeys << "\n"; 
    std::cout << "numWrites: " << numWrites << "\n"; 
    std::cout << "keyLength: " << keyLength << "\n"; 
    std::cout << "valueLength: " << valueLength << "\n"; 
    //sstWriter(sortedKeys, sortedValues, numWrites, keyLength, valueLength, NTHREADS);
    sstWriter(sortedKeys, sortedValues, numWrites, keyLength, valueLength, 128, db, bCache, cache);
    auto sst_setup_time = (TIME_NOW - start).count(); 
    std::cout << "sst_setup_time: " << sst_setup_time/1000000.0 << "\n"; 
    
    if (putsWithValues) {
        MemtableWithValues *active; 
        cudaMallocManaged(&active, sizeof(MemtableWithValues)); 
        active->size = writeCommands.size();
        sortPutsOnGPUWithValues(putCommand, writeCommands.size(), keyLength, active, batchID, valueLength);  
    }
    
}

void Batch::processDeletes()
{
    if(!deleteCommand.numDeletes)
        return; 
    activeMemtable->size = deleteCommand.numDeletes;
    activeMemtable->batchID = batchID; 
    //numWrites = writeCommands.size(); 
    sortPutsOnGPU(putCommand, writeCommands.size(), keyLength, activeMemtable, batchID, cache);  
    auto start = TIME_NOW;
    char *sortedKeys, **sortedValues; 
    sortedKeys = (char*)malloc(numWrites * keyLength); 
    sortedValues = (char**)malloc(numWrites * sizeof(char*)); 
    cudaHostRegister(sortedKeys, numWrites * keyLength, 0); 
    cudaHostRegister(sortedValues, numWrites * sizeof(char*), 0); 
    cudaMemcpy(sortedKeys, activeMemtable->d_sortedKeys, numWrites * keyLength, cudaMemcpyDeviceToHost); 
    cudaMemcpy(sortedValues, activeMemtable->d_sortedValuePointers, numWrites * sizeof(char*), cudaMemcpyDeviceToHost); 
    cudaError_t err = cudaPeekAtLastError();
    printf("Error %d cudaPeekError\n", err); 
    std::cout << "sorted keys: " << sortedKeys << "\n"; 
    std::cout << "numWrites: " << numWrites << "\n"; 
    std::cout << "keyLength: " << keyLength << "\n"; 
    std::cout << "valueLength: " << valueLength << "\n"; 
    //sstWriter(sortedKeys, sortedValues, numWrites, keyLength, valueLength, NTHREADS);
    //nothing(sortedKeys, sortedValues, numWrites, keyLength, valueLength, NTHREADS, db);
    auto sst_setup_time = (TIME_NOW - start).count(); 
    std::cout << "sst_setup_time: " << sst_setup_time/1000000.0 << "\n"; 
}

void Batch::Exit() 
{
    batchID++; 
    Execute(); 
    return; 
}


void Batch::processGets()
{
    std::cout << numReads << "\n"; 
    if (!numReads)
        return; 
    char** getValuePointers = nullptr;

        //immutableMemtable = activeMemtable; 
    searchGetsOnGPU(getCommand.keys, activeMemtable, immutableMemtable, keyLength, numReads, getValuePointers, getCommand.operationIDs, cache); 
        //cpu_get(getCommand.keys, getCommand.values, keyLength, valueLength); 
}


void Batch::processRange()
{
    if (!rangeCommand.numRangeQs)
        return; 
    char** rangeValuePointers;
    //immutableMemtable = activeMemtable; 
    immutableMemtable->size = 0;
    rangeOnGPU(rangeCommand.startKeys, rangeCommand.endKeys, activeMemtable, immutableMemtable, keyLength, rangeCommand.numRangeQs, rangeValuePointers, rangeCommand.operationIDs); 
}

void Batch::processUpdates()
{
    if (!numUpdates)
        return; 

    mergeCommand.keyLength = keyLength;
    mergeCommand.valueLength = valueLength;
    std::cout << "key length: " << keyLength << "\n"; 
    mergeCommand.numMerges = numUpdates; 
    std::cout << "num updates: " << mergeCommand.numMerges << "\n"; 
    cudaMallocManaged((void**)&immutableMemtable, sizeof(Memtable)); 
    immutableMemtable = activeMemtable; 
    cudaMallocManaged((void**)&activeMemtable, sizeof(Memtable)); 
    activeMemtable->batchID = batchID; 
    gpuIncrements(activeMemtable, immutableMemtable, numUpdates, mergeCommand, batchID, cache); 
}
