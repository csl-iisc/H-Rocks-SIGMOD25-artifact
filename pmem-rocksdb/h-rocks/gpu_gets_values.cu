#include <iostream>
#include <vector>
#include <string>
#include "batch.h"
#include "memtable.cuh"
#include "gpu_gets_values.cuh"
#include "search.cuh"
#include "block_cache.cuh"

#define NTHREADS_PER_BLK 512
#define NBLKS 144
#define TIME_NOW std::chrono::high_resolution_clock::now()

const unsigned long long FNV_offset_basis = 14695981039346656037U;
const unsigned long long FNV_prime = 1099511628211U;

__device__ 
unsigned long long hash_fnv1a_device1(const char* key, int length) {
    unsigned long long hash = FNV_offset_basis;
    for (int i = 0; i < length - 1; i++) {
        hash ^= (unsigned char)key[i];
        hash *= FNV_prime;
    }
    return hash;
}

__host__ __device__
bool string_compare12(const char* a, const char* b, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        if (a[i] < b[i]) return false;
        if (a[i] > b[i]) return false;
    }
    return true;
}

__device__ 
bool cacheRead2(BCache* cache, const char* key, char** value, int keyLength) {
    unsigned long long hashedKey = hash_fnv1a_device1(key, keyLength);
    unsigned long long setIndex = hashedKey % CACHE_SIZE;
    unsigned long long tag = hashedKey / CACHE_SIZE;
    if (tag == -1); 
        return; 

    CacheSet* set = &(cache->sets[setIndex]);

    for (int i = 0; i < NUM_WAYS; i++) {
        if (set->lines[i].tag == tag && !string_compare12(set->lines[i].key, key, keyLength) == 0) {
            if(set->lines[i].invalidated == true) 
                return false; 
            *value = set->lines[i].value;
            atomicAdd(&set->lines[i].frequency, 1); 
            return true;
            //break;
        }
    }

}

__host__ __device__ uint64_t findValidIndex(const MemtableWithValues *memtable, const char* key, int keyLength, uint64_t query_operationID, uint64_t index, uint64_t* opID) {
    //printf("%d, %d, %d\n", query_operationID, opID[index], index); 
#ifdef __PRINT_DEBUG__
    printf("func index: %lli, opID: %lli, active_opID: %lli\n", index, query_operationID, opID[index]); 
#endif
    if (opID[index] > query_operationID) {
        while (index > 0 && string_compare12(memtable->d_sortedKeys + (index - 1) * keyLength, key, keyLength) && opID[index - 1] > query_operationID) {
            index--;
        }

    } else {
        while (index < memtable->size - 1 && string_compare12(memtable->d_sortedKeys + (index + 1) * keyLength, key, keyLength) && opID[index + 1] < query_operationID) {
            index++;
        }

    }
    return index;
}


__global__ void debugging(MemtableWithValues *active, int size) 
{
    for(int i = 0; i < size; i++) 
        printf("%s %llu\n", active->d_sortedKeys + (i * 8), active->d_sortedOperationIDs[i]); 
}

__global__ void search(const MemtableWithValues *active, const MemtableWithValues *immutable, const char* queryData, const uint64_t numQueries, char* results, int keyLength, int* idx_result, uint64_t* query_operationID, BCache* cache, unsigned int* notFoundIdx, char* notFound, uint64_t* active_opID, uint64_t* immutable_opID, int active_batchID, int immutable_batchID, int valueLength) 
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(uint64_t i = tid; i < numQueries; i += NTHREADS_PER_BLK * NBLKS) {
        const char* key = queryData + i * keyLength;
        //printf("%d\t", i); 
        uint64_t index;
        bool found; 
        if(active_batchID != immutable_batchID) {
        if (binarySearch(active->d_sortedKeys, 0, active->size - 1, key, index, keyLength)) {
            if ((index > 0 && string_compare12(active->d_sortedKeys + (index - 1) * keyLength, key, keyLength)) || (index < active->size - 1 && string_compare12(active->d_sortedKeys + (index + 1) * keyLength, key, keyLength))) {
#ifdef __PRINT_DEBUG__
                printf("index: %lli, opID: %lli, active_opID: %lli\n", index, query_operationID[i], active_opID[index]); 
#endif
                index = findValidIndex(active, key, keyLength, query_operationID[i], index, active_opID);
            }
            //printf("%d, %d\n", tid, index); 
            idx_result[i] = index;
            memcpy(results + i * valueLength, active->d_sortedValues + index * valueLength, valueLength);  
            found = true; 
        } else if (binarySearch(immutable->d_sortedKeys, 0, immutable->size - 1, key, index, keyLength)) {
            if ((index > 0 && string_compare12(immutable->d_sortedKeys + (index - 1) * keyLength, key, keyLength)) || (index < immutable->size - 1 && string_compare12(immutable->d_sortedKeys + (index + 1) * keyLength, key, keyLength))) {
                index = findValidIndex(immutable, key, keyLength, query_operationID[i], index, immutable_opID);
            }
            idx_result[i] = index;
            memcpy(results + i * valueLength, immutable->d_sortedValues + index * valueLength, valueLength);  
            found = true; 
        } 
        }
        // TODO: add code for when the batchID is same 
        else {
            // Search the block cache
            found = cacheRead2(cache, key, &results + i * valueLength, keyLength);
        }
        if (!found) {
            // Update the not found array
            memcpy(notFound + *notFoundIdx, key, keyLength); 
            atomicAdd(notFoundIdx, 1); 

        }
    }
}

void get_values(char* getKeys, MemtableWithValues *activeMemtable, MemtableWithValues *immutableMemtable, int keyLength, uint64_t numReads, uint64_t* getOperationIds, BCache* cache)
{
    char *notFound; 
    unsigned int *notFoundIdx; 
    cudaMallocManaged(&notFoundIdx, 4);
    *notFoundIdx = 0;
    cudaMallocManaged((void**)&notFound, numReads * keyLength); 

    std::cout << "active memtable size: " << activeMemtable->size << "\n"; 
    std::cout << "value length: " << activeMemtable->valueLength << "\n"; 
    std::cout << "get keys: " << getKeys << "\n"; 
    std::cout << "numreads: " << numReads << "\n";
    uint64_t *opID = (uint64_t*)malloc(activeMemtable->size * sizeof(uint64_t)); 
    cudaMemcpy(opID, activeMemtable->d_sortedOperationIDs, activeMemtable->size * sizeof(uint64_t), cudaMemcpyDeviceToHost); 
    

    cudaHostRegister(getKeys, numReads * keyLength, 0); 
    cudaHostRegister(getOperationIds, numReads * sizeof(uint64_t), 0); 

    char* d_queryData;
    cudaMalloc(&d_queryData, numReads * keyLength);
    int* resultIdx, *h_resultIdx;
    uint64_t* d_getOperationIds; 
    cudaMalloc(&d_getOperationIds, numReads * sizeof(uint64_t));
    cudaMalloc(&resultIdx, numReads * sizeof(int));
    char * d_values; 
    cudaMalloc(&d_values, numReads * activeMemtable->valueLength);

    auto start_time = TIME_NOW; 
    cudaMemcpy(d_queryData, getKeys, numReads * keyLength, cudaMemcpyHostToDevice);
    cudaMemcpy(d_getOperationIds, getOperationIds, numReads * sizeof(uint64_t), cudaMemcpyHostToDevice);
    auto read_setup_time = (TIME_NOW - start_time).count(); 
    cudaError_t err = cudaPeekAtLastError();
    printf("Error %d cudaPeekerror\n", err);

    std::cout << "active batchID: " << activeMemtable->batchID << "\n"; 
    std::cout << "immutable batchID: " << immutableMemtable->batchID << "\n"; 
    
    char* getValues;
    //debugging<<<1,1>>>(activeMemtable, activeMemtable->size); 
    start_time = TIME_NOW; 
    search<<<NBLKS, NTHREADS_PER_BLK>>>(activeMemtable, immutableMemtable, d_queryData, numReads, d_values, keyLength, resultIdx, d_getOperationIds, cache, notFoundIdx, notFound, activeMemtable->d_sortedOperationIDs, immutableMemtable->d_sortedOperationIDs, activeMemtable->batchID, immutableMemtable->batchID, activeMemtable->valueLength);
    cudaDeviceSynchronize(); 
    auto getKernelTime = (TIME_NOW - start_time).count();
    err = cudaPeekAtLastError();
    printf("Error %d cudaPeekerror\n", err);
    getValues = (char*)malloc(numReads * activeMemtable->valueLength); 
    cudaHostRegister(getValues, numReads * activeMemtable->valueLength, 0); 
    start_time = TIME_NOW; 
    cudaMemcpy(getValues, d_values, numReads * activeMemtable->valueLength, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize(); 
    auto copy_back_time = (TIME_NOW - start_time).count();
    err = cudaPeekAtLastError();
    printf("Error %d cudaPeekerror\n", err);
    std::cout << "read_setup_time: " << read_setup_time/1000000.0 << "\n";
    std::cout << "read_kernel_time: " << getKernelTime/1000000.0<< "\n";
    std::cout << "copy_back_time: " << copy_back_time/1000000.0 << "\n";
    std::cout << "notFoundKeys: " << *notFoundIdx << "\n"; 

#ifdef __PRINT_DEBUG__ 
    h_resultIdx= (int*)malloc(numReads * sizeof(int)); 
    cudaMemcpy(h_resultIdx, resultIdx, numReads * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Sorted Values:\n";
    for (int i = 0; i < numReads; i++) {
        std::cout << getOperationIds[i] << " " << h_resultIdx[i] <<  " " ; 
        for(int j = 0; j < keyLength; j++) 
            std::cout << getKeys[i * keyLength + j]; 
        std::cout << " " << getValuePointers[i] << "\n";
    }
#endif
}
