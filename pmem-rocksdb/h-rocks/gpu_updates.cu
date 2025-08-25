#include <iostream>
#include "gpu_updates.cuh"
#include "memtable.cuh"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include "batch.h"
#include <chrono>
#include "libgpm.cuh"
#include "gpm-helper.cuh"
#include "libgpmlog.cuh"
#include "string.h"
#include "search.cuh"
#include "block_cache.cuh"

#define NTHREADS_PER_BLK 512
#define NBLKS 144 

#define TIME_NOW std::chrono::high_resolution_clock::now()

__host__ __device__
int string_cmp2(const char* a, const char* b, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

typedef struct log_entry_s {
    char key[8]; 
    uint64_t sequenceId; 
    char* value; 
} log_entry_t; 

__host__ __device__
bool string_compare3(const char* a, const char* b, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        if (a[i] < b[i]) return true;
        if (a[i] > b[i]) return false;
    }
    return false;
}

// Values will be updated by the other kernel

__global__ void updateIndicesKernel(Memtable* table, Memtable* temp, int size, unsigned long long int* d_sortedIndices, int keyLength, char* d_tempStr, char* sortedStr, gpmlog* dlog, uint64_t *d_temp_idx) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(uint64_t i = idx; i < size; i += NTHREADS_PER_BLK * NBLKS) {
        int sorted_index = d_sortedIndices[idx]; 
        log_entry_t entry; 
        // Update the indices of putValuePointers and putOperationIDs based on sortedIndices
        memcpy(sortedStr + idx * keyLength, d_tempStr + sorted_index * keyLength, keyLength - 1);
        memcpy(entry.key, sortedStr + idx * keyLength, keyLength - 1);

        table->d_sortedOperationIDs[idx] = d_temp_idx[sorted_index];
        entry.sequenceId = d_temp_idx[sorted_index];
        gpmlog_insert(dlog, &entry, sizeof(log_entry_t), idx); 
    }
}

__device__ void intToString(int input, char *output, int numDigits) {
    int i = numDigits - 1;
    while (input > 0 && i >= 0) {
        output[i--] = '0' + (input % 10); // Add digit to output string
        input /= 10;
    }
    while (i >= 0) {
        output[i--] = '0'; // Add leading zeros to output string
    }
    output[numDigits] = '\0'; // Add null terminator to output string
#ifdef __PRINT_DEBUG__
    printf("%s\t", output); 
#endif
}

__device__ int stringToInt(char *input, int valueLength) {
    int output = 0; 
    if(input == NULL)
        return 0; 
#ifdef __PRINT_DEBUG__
    printf("TID: %d input: %s valueLen: %d\n", threadIdx.x, input, valueLength); 
#endif
    for(int i = 0; i < valueLength - 1; ++i) {
        if (*(input + i) >= '0' && *(input + i) <= '9') {
            output = (output * 10) + (input[i] - '0'); // Convert digit to integer
        } else {
            break; // Stop processing if non-digit character encountered
        }
    }
    return output; 
}


struct string_comparator {
    const char* data;
    size_t length;
    string_comparator(const char* data, size_t length) : data(data), length(length) {}
    __host__ __device__
        bool operator()(size_t a_idx, size_t b_idx) const {
            const char* a = data + a_idx * length;
            const char* b = data + b_idx * length;
            return string_compare3(a, b, length);
        }
};


__device__ __forceinline__ 
    void
incrementKey(Memtable *immutable, uint64_t immutableIdx, Memtable *uTable, uint64_t idxToUpdate, gpmlog *dlog, char* valueBuffer, uint32_t keyLength, uint32_t valueLength, uint64_t uTable_size, char* uTable_d_sortedKeys) 
{
    const char* key = uTable->d_sortedKeys + idxToUpdate * keyLength; 

    // Updated value added in updatememt

    char *rightNeighbor, newValue[8]; 
    //char **valuePtrPtr  = immutable->d_sortedValuePointers + immutableIdx; 
    char *valuePtr = immutable->d_sortedValuePointers[immutableIdx]; 
    //char *valuePtr = *valuePtrPtr; 
    int value = 0; 
    /*
    value = stringToInt(valuePtr, valueLength); 
    value++;
    intToString(value, newValue, valueLength);
    // Memcpy newvalue to valuebuffer which is on UVA 
    memcpy(valueBuffer + idxToUpdate * valueLength, newValue, valueLength); 

    // Update the memtable to point to the value pointer
    uTable->d_sortedValuePointers[idxToUpdate] = valueBuffer + idxToUpdate * valueLength; 
    */

#ifdef __PRINT_DEBUG__ 
    printf("key: %s\t", key); 
    printf("idx: %d valuePtr:%s\t", immutableIdx, valuePtr); 
#endif
       do {
       value = stringToInt(valuePtr, valueLength); 
       value++;
       intToString(value, newValue, valueLength);
    // Memcpy newvalue to valuebuffer which is on UVA 
    memcpy(valueBuffer + idxToUpdate * valueLength, newValue, valueLength); 

    // Update the memtable to point to the value pointer
    uTable->d_sortedValuePointers[idxToUpdate] = valueBuffer + idxToUpdate * valueLength; 

    if(idxToUpdate >= uTable_size) 
    return; 

    rightNeighbor = uTable_d_sortedKeys + (idxToUpdate + 1) * keyLength; 
    if (rightNeighbor == NULL)
    return; 

    if(uTable->d_sortedOperationIDs[idxToUpdate + 1] < immutable->d_sortedOperationIDs[immutableIdx + 1]) {
    valuePtr = uTable->d_sortedValuePointers[idxToUpdate]; 
    } else {
    valuePtr = immutable->d_sortedValuePointers[immutableIdx + 1]; 
    immutableIdx++; 
    }
    idxToUpdate++;
    } while(!string_compare3(key, rightNeighbor, keyLength) && (idxToUpdate < uTable_size)); 
}


    __global__
void incrementKernel(Memtable *activeTable, Memtable *immutableTable, uint64_t numUpdates, char *notFound, unsigned int *notFoundIdx, gpmlog* dlog, char* valueBuffer, uint32_t keyLength, uint64_t activeTable_size, uint64_t immutable_size, uint32_t valueLength, char* immutable_d_sortedKeys, char* activeTable_d_sortedKeys) 
{
    // Find the key to update
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(uint64_t i = idx; i < activeTable_size; i += NTHREADS_PER_BLK * NBLKS) {
        const char* key = activeTable_d_sortedKeys + i * keyLength; 
#ifdef __PRINT_DEBUG__
        printf("key:%s\t", key); 
#endif 
        uint64_t index = 0;     
        if(i != 0) {
            const char* leftNeighbor = activeTable_d_sortedKeys + (i - 1) * keyLength; 
            if (leftNeighbor != "" && string_compare3(key, leftNeighbor, keyLength) == 0)
                return; 
        }
        
#ifdef __PRINT_DEBUG__
        printf("immutable size:%d\t", immutable_size); 
#endif 
        if (binarySearch(immutable_d_sortedKeys, 0, immutable_size - 1, key, index, keyLength)) {
#ifdef __PRINT_DEBUG__
            printf("%s %d %d %s\n", key, index, keyLength);
#endif
            incrementKey(immutableTable, index, activeTable, i, dlog, valueBuffer, keyLength, valueLength, activeTable_size, activeTable_d_sortedKeys);
            // Update the value 
        } else {
            // Update the not found array
            memcpy(notFound + *notFoundIdx, key, keyLength); 
            atomicAdd(notFoundIdx, 1); 
        }
    }
}

void gpuIncrements(Memtable* activeMemtable, Memtable* immutableMemtable, uint64_t numUpdates, MergeCommand& mergeCommand, int batchID, BCache* cache) 
{
    std::cout << "num elems: " << numUpdates << "\n";
    std::cout << "key length: " << mergeCommand.keyLength << "\n";
    std::cout << "value length: " << mergeCommand.valueLength << "\n";

    Memtable* temp; 
    cudaMallocManaged((void**)&temp, sizeof(UpdateMemtable));

    char* valueBuffer = (char*) malloc(mergeCommand.valueLength * numUpdates); 
    cudaHostRegister(valueBuffer, mergeCommand.valueLength * numUpdates, 0); 
    //cudaHostAlloc((void**)&valueBuffer, mergeCommand.valueLength * numUpdates, 0); 

    temp->size = numUpdates;
    temp->keyLength = mergeCommand.keyLength;
    temp->valueLength = mergeCommand.valueLength;
    uint32_t keyLength = mergeCommand.keyLength;

    char *notFound; 
    unsigned int *notFoundIdx; 
    cudaMallocManaged(&notFoundIdx, 4);
    *notFoundIdx = 0;


    auto start_time = TIME_NOW; 
    cudaMalloc((void**)&temp->d_sortedKeys, numUpdates * mergeCommand.keyLength); 
    cudaMallocManaged((void**)&notFound, numUpdates * mergeCommand.keyLength); 
    cudaMalloc((void**)&temp->d_sortedOperationIDs, numUpdates * sizeof(uint64_t)); 
    cudaError_t err = cudaPeekAtLastError();
    printf("Error %d cudaPeekerror\n", err);

    cudaMemcpy(temp->d_sortedKeys, mergeCommand.keys, numUpdates * mergeCommand.keyLength, cudaMemcpyHostToDevice); 
    cudaMemcpy(temp->d_sortedOperationIDs, mergeCommand.operationIDs, numUpdates * sizeof(uint64_t), cudaMemcpyHostToDevice); 
    err = cudaPeekAtLastError();
    printf("Error %d cudaPeekerror\n", err);

    activeMemtable->size = numUpdates; 
    activeMemtable->keyLength = mergeCommand.keyLength;
    activeMemtable->valueLength = mergeCommand.valueLength;
    activeMemtable->size = numUpdates; 
    cudaMalloc((void**)&activeMemtable->d_sortedKeys, numUpdates * mergeCommand.keyLength); 
    cudaMalloc((void**)&activeMemtable->d_sortedValuePointers, numUpdates * mergeCommand.valueLength); 
    cudaMalloc((void**)&activeMemtable->d_sortedOperationIDs, numUpdates * sizeof(uint64_t)); 

    cudaMemcpy(activeMemtable->d_sortedKeys, mergeCommand.keys, numUpdates * mergeCommand.keyLength, cudaMemcpyHostToDevice); 
    cudaMemcpy(activeMemtable->d_sortedOperationIDs, mergeCommand.operationIDs, numUpdates * sizeof(uint64_t), cudaMemcpyHostToDevice); 
    auto updateSetupTime = (TIME_NOW - start_time).count();

    err = cudaPeekAtLastError();
    printf("Error %d cudaPeekerror\n", err);

    std::cout << "num elems: " << temp->size << "\n";
    std::cout << "active memtable: " << activeMemtable->size << "\n";

    err = cudaPeekAtLastError();
    printf("Error %d cudaPeekerror\n", err);


    cudaDeviceSynchronize(); 
    start_time = TIME_NOW; 
    thrust::device_vector<char> d_strings(mergeCommand.keyLength * numUpdates); 
    cudaMemcpy(d_strings.data().get(), activeMemtable->d_sortedKeys, numUpdates * keyLength, cudaMemcpyHostToDevice);

    // Generate an array of indices
    thrust::device_vector<unsigned long long int> indices(numUpdates);
    thrust::sequence(indices.begin(), indices.end());
    //auto setupTime = (TIME_NOW - start).count();
    auto sortSetupTime = (TIME_NOW - start_time).count();

    err = cudaPeekAtLastError();
    printf("Error %d cudaPeekerror\n", err);


    char  *sortedStr, *d_tempStr; 
    cudaMalloc((void**)&sortedStr, numUpdates * keyLength);
    cudaMalloc((void**)&d_tempStr, numUpdates * keyLength);
    err = cudaPeekAtLastError();
    printf("Error %d cudaPeekerror\n", err);
    auto start = TIME_NOW;
    thrust::sort(
            thrust::device,
            indices.begin(),
            indices.end(),
            string_comparator(thrust::raw_pointer_cast(d_strings.data()), keyLength)
            );

    unsigned long long int* d_indices = thrust::raw_pointer_cast(indices.data());
    auto sortTime = (TIME_NOW - start).count();


    // Update values. 

    err = cudaPeekAtLastError();
    printf("Error %d cudaPeekerror\n", err);

    // Rearrange the value pointers and operation IDs based on the sorted index array
    int gridSize = (numUpdates + NTHREADS_PER_BLK - 1) / NTHREADS_PER_BLK;

    size_t logSize = (numUpdates + 1) * sizeof(log_entry_t);
    std::cout << "size of log: " << logSize << "\n"; 
    batchID++; 
    std::string logFileName = "rdb_log" + std::to_string(batchID);
    char* fileName = new char[logFileName.length() + 1]; // Allocate memory for the string
    strcpy(fileName, logFileName.c_str());
    cout << "Filename: " << fileName << "\n"; 
    gpmlog *dlog = gpmlog_create_managed(fileName, logSize, gridSize + 1, NTHREADS_PER_BLK); 

    auto startTime = TIME_NOW; 
    updateIndicesKernel<<<NBLKS, NTHREADS_PER_BLK>>>(temp, activeMemtable, numUpdates, d_indices, keyLength, d_tempStr, sortedStr, dlog, temp->d_sortedOperationIDs);
    cudaDeviceSynchronize(); 
    auto memtableTime = (TIME_NOW - startTime).count();  
    err = cudaPeekAtLastError();
    printf("Error %d cudaPeekerror\n", err);
    startTime - TIME_NOW; 
    incrementKernel<<<NTHREADS_PER_BLK, NBLKS>>>(activeMemtable, immutableMemtable, numUpdates, notFound, notFoundIdx, dlog, valueBuffer, keyLength, activeMemtable->size, immutableMemtable->size, immutableMemtable->valueLength, immutableMemtable->d_sortedKeys, activeMemtable->d_sortedKeys); 
    cudaDeviceSynchronize(); 
    auto incrementTime = (TIME_NOW - startTime).count();  
    err = cudaPeekAtLastError();
    printf("Error %d cudaPeekerror\n", err);

#ifdef __PRINT_DEBUG__ 
    h_resultIdx= (int*)malloc(numUpdates * sizeof(int)); 
    cudaMemcpy(h_resultIdx, d_indices, numReads * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Sorted Values:\n";
    for (int i = 0; i < numReads; i++) {
        std::cout << getOperationIds[i] << " " << h_resultIdx[i] <<  " " ; 
        for(int j = 0; j < keyLength; j++) 
            std::cout << getKeys[i * keyLength + j]; 
        std::cout << " " << getValuePointers[i] << "\n";
    }
#endif

    cout << "sortTime: " << sortTime/1000000.0 << "\n"; 
    cout << "sortSetupTime: " << sortSetupTime/1000000.0 << "\n"; 
    cout << "updateSetupTime: " << updateSetupTime/1000000.0 << "\n"; 
    cout << "memtableTime: " << memtableTime/1000000.0 << "\n"; 
    cout << "incrementTime: " << incrementTime/1000000.0 << "\n"; 

    return; 
}
