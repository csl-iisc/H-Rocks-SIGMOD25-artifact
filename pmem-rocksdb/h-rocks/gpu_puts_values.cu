#include <iostream>
#include "gpu_puts_values.cuh"
#include "memtable.cuh"
//#include <cub/cub.cuh>
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

#define NTHREADS_PER_BLK 512
#define NBLKS 144

#define TIME_NOW std::chrono::high_resolution_clock::now()
struct StringCompare
{
    __host__ __device__ bool operator()(const char *a, const char *b) const
    {
        return strcmp(a, b) < 0;
    }
};

typedef struct log_entry_s {
    char key[8]; 
    char value[8]; 
    uint64_t sequenceId; 
} log_entry_t; 

typedef struct sstFile_t {
    char key[8]; 
    char value[8]; 
    uint64_t keyLen; 
    uint64_t valueLen; 
    uint64_t opID; 
} sstFile; 

__host__ __device__
bool string_compareV(const char* a, const char* b, size_t length) {
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
            return string_compareV(a, b, length);
        }
};

__device__ void string_copyV(const char* src, char* dst, const int length)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < length) {
        dst[tid] = src[tid];
    }
}


// GPU kernel function for updating the sorted indices
__global__ void updateIndicesKernel(MemtableWithValues *table, MemtableWithValues *temp, int size, int* d_sortedIndices, int keyLength, char* d_tempStr, char* sortedStr, gpmlog* dlog, char* d_tempValue, unsigned int valueLength) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(uint64_t i = idx; i < size; i += NTHREADS_PER_BLK * NBLKS) {
        int sorted_index = d_sortedIndices[i]; 
        log_entry_t entry; 
        // Update the indices of putValuePointers and putOperationIDs based on sortedIndices
        memcpy(sortedStr + i * keyLength, d_tempStr + sorted_index * keyLength, keyLength - 1);
        memcpy(table->d_sortedValues + i * valueLength, d_tempValue + sorted_index * valueLength , valueLength - 1);
        memcpy(entry.key, sortedStr + i * keyLength, keyLength - 1);
        memcpy(entry.value, table->d_sortedValues + i * valueLength, valueLength - 1);

        table->d_sortedOperationIDs[i] = temp->d_sortedOperationIDs[sorted_index];
        entry.sequenceId = temp->d_sortedOperationIDs[sorted_index];
        gpmlog_insert(dlog, &entry, sizeof(log_entry_t), i); 
    }
}

__global__ void generateSstFile(MemtableWithValues *table, sstFile* writeSstFile, uint64_t size) 
{
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int keyLength = 8, valueLength = 8;
    for(uint64_t i = idx; i < size; i += NTHREADS_PER_BLK * NBLKS) {
        memcpy(writeSstFile[i].key, table->d_sortedKeys + i * keyLength, keyLength); 
        memcpy(writeSstFile[i].value, table->d_sortedValues + i * valueLength, valueLength); 
        writeSstFile[i].opID = table->d_sortedOperationIDs[i]; 
        writeSstFile[i].keyLen = keyLength; 
        writeSstFile[i].valueLen = valueLength; 
        gpm_persist(writeSstFile + i, sizeof(sstFile)); 
    }
}

//void sortPutsOnGPU(char *putKeys, char **putValuePointers, uint64_t *putOperationIDs, int num_elems, int keyLength, Memtable& table, int batchID) 
void sortPutsOnGPUWithValues(PutCommand &putCommand, uint64_t num_elems, uint64_t keyLength, MemtableWithValues *table, uint64_t batchID, int valueLength)
{
    cudaFree(0);
    std::cout << "Size of batch: " << num_elems << "\n"; 
    batchID++; 
    table->batchID = batchID; 

    MemtableWithValues *temp; 
    cudaMallocManaged(&temp, sizeof(MemtableWithValues)); 
    temp->size = table->size;  
    auto start = TIME_NOW; 
    cudaMalloc((void**)&table->d_sortedKeys, num_elems * keyLength);
    cudaMalloc((void**)&table->d_sortedValues, num_elems * valueLength);
    cudaMalloc((void**)&table->d_sortedOperationIDs, num_elems * sizeof(uint64_t));

    char *d_tempStr, *sortedStr, *d_tempValue; 
    //cudaMalloc((void**)&temp.d_sortedKeys, num_elems * keyLength);
    cudaMalloc((void**)&d_tempStr, num_elems * keyLength);
    cudaMalloc((void**)&sortedStr, num_elems * keyLength);
    cudaMalloc((void**)&d_tempValue, num_elems * valueLength);
    cudaMalloc((void**)&temp->d_sortedOperationIDs, num_elems * sizeof(uint64_t));
    cudaMalloc((void**)&temp->d_sortedValues, num_elems * valueLength);

    // Copy input arrays to corresponding device arrays inside memtable
    cudaMemcpy(d_tempStr, putCommand.keys, num_elems * keyLength, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tempValue, putCommand.values, num_elems * valueLength, cudaMemcpyHostToDevice);
    cudaMemcpy(temp->d_sortedOperationIDs, putCommand.operationIDs, num_elems * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(temp->d_sortedValues, putCommand.values, num_elems * valueLength, cudaMemcpyHostToDevice);

    // Create thrust device pointers for sorted keys and operation IDs
    thrust::device_vector<char> d_strings(keyLength * num_elems); 
    cudaMemcpy(d_strings.data().get(), putCommand.keys, num_elems * keyLength, cudaMemcpyHostToDevice);

    // Generate an array of indices
    thrust::device_vector<int> indices(num_elems);
    thrust::sequence(indices.begin(), indices.end());
    cudaDeviceSynchronize(); 
    auto setupTime = (TIME_NOW - start).count();
    cudaError_t err = cudaPeekAtLastError();
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

    size_t logSize = (num_elems + 1) * sizeof(log_entry_t);
    std::cout << "size of log: " << logSize << "\n"; 
    std::string logFileName = "rdb_log" + std::to_string(batchID);
    char* fileName = new char[logFileName.length() + 1]; // Allocate memory for the string
    strcpy(fileName, logFileName.c_str());
    cout << "Filename: " << fileName << "\n"; 
    gpmlog *dlog = gpmlog_create_managed(fileName, logSize, gridSize + 1, NTHREADS_PER_BLK); 
    std::cout << "Value length: " << valueLength; 

    start = TIME_NOW; 

    updateIndicesKernel<<<NBLKS, NTHREADS_PER_BLK>>>(table, temp, num_elems, d_indices, keyLength, d_tempStr, sortedStr, dlog, d_tempValue, valueLength);
    cudaDeviceSynchronize(); 
    auto memtableTime = (TIME_NOW - start).count();

    err = cudaPeekAtLastError();
    printf("Error %d cudaPeekerror\n", err);
    cudaMemcpy(table->d_sortedKeys, sortedStr, num_elems * keyLength * sizeof(char), cudaMemcpyDeviceToDevice);
    err = cudaPeekAtLastError();
    printf("Error %d cudaPeekerror\n", err);

    sstFile* writeSstFile; 
    size_t len = sizeof(sstFile) * num_elems; 
    const char* path = "sst_file.dat";
    writeSstFile = (sstFile*) gpm_map_file(path, len, true); 

    start = TIME_NOW; 
    generateSstFile<<<NBLKS, NTHREADS_PER_BLK>>>(table, writeSstFile, num_elems); 
    cudaDeviceSynchronize(); 
    auto sstFileTime = (TIME_NOW - start).count();

    std::cout << "setup_time: " << setupTime/1000000.0 << "\n";
    std::cout << "sort_time: " << sortTime/1000000.0 << "\n";
    std::cout << "memtable_time: " << memtableTime/1000000.0 << "\n";
    std::cout << "sstFileTime: " << sstFileTime/1000000.0 << "\n";

#ifdef __PRINT_DEBUG__
    char* sorted_keys = new char[num_elems * keyLength];
    uint64_t* sorted_operationIDs = new uint64_t[num_elems];
    char** sorted_putValuePts = (char**) malloc(sizeof(char*) * num_elems); 
    cudaMemcpy(sorted_keys, table.d_sortedKeys, num_elems * keyLength * sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(sorted_operationIDs, table.d_sortedOperationIDs, num_elems * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(sorted_putValuePts, table.d_sortedValuePointers, num_elems * sizeof(char*), cudaMemcpyDeviceToHost);
    std::cout << "In GPU: " << num_elems << " " << sorted_keys << "\n"; 
    thrust::host_vector<int> h_indices(indices); 
    std::cout << "Sorted Keys:\n";
    for (int i = 0; i < num_elems; i++) {
        std::cout << "i: " << h_indices[i] << " "; 
        for (int j = 0; j < keyLength; j++) {
            std::cout << sorted_keys[i * keyLength + j];
        }
        std::cout << " " << sorted_putValuePts[i];
        std::cout << "\n";
    }
    std::cout << "Sorted Values:\n";
    for (int i = 0; i < num_elems; i++) {
        std::cout << sorted_putValuePts[i];
        std::cout << "\n";
    }
#endif

    /*
    // Free the GPU memory
    cudaFree(d_putKeys);
    cudaFree(d_sortedIndices);
    cudaFree(d_tempStorage);
     */
    /*
       cudaFree(temp.d_sortedValuePointers); 
       cudaFree(temp.d_sortedKeys); 
       cudaFree(temp.d_sortedOperationIDs); 
       cudaFree(sortedStr); 
       cudaFree(d_tempStr); 
     */
}

