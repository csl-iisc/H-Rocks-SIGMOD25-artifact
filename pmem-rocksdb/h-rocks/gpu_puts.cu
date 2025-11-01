#include <iostream>
#include "gpu_puts.cuh"
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
#include "block_cache.cuh"

#define NTHREADS_PER_BLK 512
#define NBLKS 144

#define TIME_NOW std::chrono::high_resolution_clock::now()

struct StringCompare
{
    __host__ __device__ __forceinline__ bool operator()(const char *a, const char *b) const
    {
        return strcmp(a, b) < 0;
    }
};

__host__ __device__
bool inline string_compare(const char* a, const char* b, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        if (a[i] < b[i]) return true;
        if (a[i] > b[i]) return false;
    }
    return false;
}

struct string_comparator {
    const char* data;
    size_t length;
    string_comparator(const char* data, size_t length) : data(data), length(length) {}
    __host__ __device__
        bool operator()(size_t a_idx, size_t b_idx) const {
            const char* a = data + a_idx * length;
            const char* b = data + b_idx * length;
            return string_compare(a, b, length);
        }
};

__device__ void string_copy(const char* src, char* dst, const int length)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < length) {
        dst[tid] = src[tid];
    }
}

__device__ void stringCpy(char* dst, const char* src, const int length)
{
    for(int i = 0; i < length; ++i) {
        dst[i] = src[i];
    }
}

bool string_compare_bool(const char* a, const char* b, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        if (a[i] < b[i]) return false;
        if (a[i] > b[i]) return false;
    }
    return true;
}



// GPU kernel function for updating the sorted indices
__global__ void updateIndicesKernel(Memtable *table, Memtable *temp, int size, int* d_sortedIndices, int keyLength, char* d_tempStr, char* sortedStr, BCache* cache, uint64_t* seqIdLog, char** valueLog, char* keyLog) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(uint64_t i = idx; i < size; i += NTHREADS_PER_BLK * NBLKS) {
        int sorted_index = d_sortedIndices[i]; 
        // Update the indices of putValuePointers and putOperationIDs based on sortedIndices
        memcpy(sortedStr + i * keyLength, d_tempStr + sorted_index * keyLength, keyLength - 1);
        cudaMemcpyAsync(keyLog + i * keyLength, sortedStr + i * keyLength, keyLength - 1, cudaMemcpyDeviceToHost);

        table->d_sortedValuePointers[i] = temp->d_sortedValuePointers[sorted_index];
        //entry.value = temp->d_sortedValuePointers[sorted_index];
        valueLog[i] = temp->d_sortedValuePointers[sorted_index];
        table->d_sortedOperationIDs[i] = temp->d_sortedOperationIDs[sorted_index];
        //entry.sequenceId = temp->d_sortedOperationIDs[sorted_index];
        seqIdLog[i]= temp->d_sortedOperationIDs[sorted_index];
        gpm_drain(); 
    }
}

//void sortPutsOnGPU(char *putKeys, char **putValuePointers, uint64_t *putOperationIDs, int num_elems, int keyLength, Memtable& table, int batchID) 
void sortPutsOnGPU(PutCommand &putCommand, uint64_t num_elems, uint64_t keyLength, Memtable *table, uint64_t batchID, BCache *cache)
{
    cudaFree(0);
    std::cout << "Size of batch: " << num_elems << "\n"; 

    Memtable* temp; 
    cudaMallocManaged(&temp, sizeof(Memtable)); 
    cudaHostRegister(putCommand.keys, num_elems * keyLength, 0); 
    cudaHostRegister(putCommand.valuePtrs, num_elems * sizeof(char*), 0); 
    cudaHostRegister(putCommand.operationIDs, num_elems * sizeof(uint64_t), 0); 
    cudaError_t err = cudaPeekAtLastError();
    printf("Error %d cudaPeekerror\n", err);

    temp->size = table->size;  
    cudaMalloc((void**)&table->d_sortedKeys, num_elems * keyLength);
    cudaMalloc((void**)&table->d_sortedValuePointers, num_elems * sizeof(char*));
    cudaMalloc((void**)&table->d_sortedOperationIDs, num_elems * sizeof(uint64_t));

    char* d_tempStr, *sortedStr; 
    //cudaMalloc((void**)&temp.d_sortedKeys, num_elems * keyLength);
    cudaMalloc((void**)&d_tempStr, num_elems * keyLength);
    cudaMalloc((void**)&sortedStr, num_elems * keyLength);
    cudaMalloc((void**)&temp->d_sortedValuePointers, num_elems * sizeof(char*));
    cudaMalloc((void**)&temp->d_sortedOperationIDs, num_elems * sizeof(uint64_t));

    // Copy input arrays to corresponding device arrays inside memtable
    //cudaMemcpy(temp.d_sortedKeys, putKeys, num_elems * keyLength, cudaMemcpyHostToDevice);
    auto start = TIME_NOW; 
    cudaMemcpy(d_tempStr, putCommand.keys, num_elems * keyLength, cudaMemcpyHostToDevice);
    //cudaMemcpy(sortedStr, putKeys, num_elems * keyLength, cudaMemcpyHostToDevice);
    cudaMemcpy(temp->d_sortedValuePointers, putCommand.valuePtrs, num_elems * sizeof(char*), cudaMemcpyHostToDevice);
    cudaMemcpy(temp->d_sortedOperationIDs, putCommand.operationIDs, num_elems * sizeof(uint64_t), cudaMemcpyHostToDevice);
    //err = cudaPeekAtLastError();
    //printf("Error %d cudaPeekerror\n", err);

    // Create thrust device pointers for sorted keys and operation IDs
    thrust::device_vector<char> d_strings(keyLength * num_elems); 
    cudaMemcpy(d_strings.data().get(), putCommand.keys, num_elems * keyLength, cudaMemcpyHostToDevice);

    // Generate an array of indices
    thrust::device_vector<int> indices(num_elems);
    thrust::sequence(indices.begin(), indices.end());
    auto setupTime = (TIME_NOW - start).count();
    err = cudaPeekAtLastError();
    printf("Error %d cudaPeekerror\n", err);


    start = TIME_NOW; 

    // Sort the array of keys and operation IDs on the device
    start = TIME_NOW;
    thrust::sort(
            thrust::device,
            indices.begin(),
            indices.end(),
            string_comparator(thrust::raw_pointer_cast(d_strings.data()), keyLength)
            );

    auto sortTime = (TIME_NOW - start).count();
    //thrust::sort_by_key(d_strings.begin(), d_strings.end(), indices.begin(), StringCompare());

    err = cudaPeekAtLastError();
    printf("Error %d cudaPeekerror\n", err);

    int* d_indices = thrust::raw_pointer_cast(indices.data());
    //table.d_sortedKeys = thrust::raw_pointer_cast(d_strings.data()); 

    // Rearrange the value pointers and operation IDs based on the sorted index array
    int gridSize = (num_elems + NTHREADS_PER_BLK - 1) / NTHREADS_PER_BLK;

    std::string logFileName = "rdb_log" + std::to_string(batchID); 
    char* fileName = new char[logFileName.length() + 1]; // Allocate memory for the string
    strcpy(fileName, logFileName.c_str());
    cout << "Filename: " << fileName << "\n"; 
    size_t sequenceIdLogSize = sizeof(uint64_t) * num_elems; 
    size_t valueLogSize = sizeof(char*) * num_elems; 
    size_t keyLogSize = keyLength * num_elems; 
    uint64_t *sequenceIdLog = (uint64_t*) gpm_map_file(fileName, sequenceIdLogSize, true); 
    char **valueLog = (char**) gpm_map_file("valueLog.dat", valueLogSize, true); 
    char *keyLog = (char*) gpm_map_file("keyLog.dat", keyLogSize, true); 
   
    start = TIME_NOW; 

    updateIndicesKernel<<<NBLKS, NTHREADS_PER_BLK>>>(table, temp, num_elems, d_indices, keyLength, d_tempStr, sortedStr, cache, sequenceIdLog, valueLog, keyLog);
    cudaDeviceSynchronize(); 

    cudaMemcpy(table->d_sortedKeys, sortedStr, num_elems * keyLength * sizeof(char), cudaMemcpyDeviceToDevice);

    auto memtableTime = (TIME_NOW - start).count();
    err = cudaPeekAtLastError();
    printf("Error %d cudaPeekerror\n", err);
    
    start = TIME_NOW; 
    pmem_mt_persist(sequenceIdLog, sequenceIdLogSize); 
    pmem_mt_persist(valueLog, valueLogSize); 
    pmem_mt_persist(keyLog, keyLogSize); 
    auto logPersistTime = (TIME_NOW - start).count();


    std::cout << "setup_time: " << setupTime/1000000.0 << "\n";
    std::cout << "sort_time: " << sortTime/1000000.0 << "\n";
    std::cout << "memtable_time: " << memtableTime/1000000.0 << "\n";
    std::cout << "log_persist_time: " << logPersistTime/1000000.0 << "\n";

#ifdef __PRINT_DEBUG__
    std::cout << num_elems << " " << keyLength <<  "\n"; 
    char* sorted_keys = new char[num_elems * keyLength];
    uint64_t* sorted_operationIDs = new uint64_t[num_elems];
    char** sorted_putValuePts = (char**) malloc(sizeof(char*) * num_elems); 
    cudaMemcpy(sorted_keys, table->d_sortedKeys, num_elems * keyLength * sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(sorted_operationIDs, table->d_sortedOperationIDs, num_elems * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(sorted_putValuePts, table->d_sortedValuePointers, num_elems * sizeof(char*), cudaMemcpyDeviceToHost);
    std::cout << "In GPU: " << num_elems << " " << sorted_keys << "\n"; 
    thrust::host_vector<int> h_indices(indices); 
    std::cout << "Sorted Keys:\n";
    for (int i = 0; i < num_elems; i++) {
        std::cout << "i: " << i << " " << h_indices[i] << " "; 
        for (int j = 0; j < keyLength; j++) {
            std::cout << sorted_keys[i * keyLength + j];
        }
        std::cout << " " << sorted_putValuePts[i];
        std::cout << "\n";
    }
    std::cout << "Sorted Values:\n";
    for (int i = 0; i < num_elems; i++) {
        std::cout << i << " " << sorted_operationIDs[i] << " " << sorted_putValuePts[i];
        std::cout << "\n";
    }
#endif

    /*
    // Free the GPU memory
    cudaFree(d_putKeys);
    cudaFree(d_sortedIndices);
    cudaFree(d_tempStorage);
       cudaFree(temp->d_sortedValuePointers); 
       cudaFree(temp->d_sortedKeys); 
       cudaFree(temp->d_sortedOperationIDs); 
       cudaFree(sortedStr); 
       cudaFree(d_tempStr); 
     */
}

