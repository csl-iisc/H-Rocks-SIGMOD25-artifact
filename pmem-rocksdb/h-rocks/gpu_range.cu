#include "memtable.cuh"
#include "command.h" 
#include "libgpm.cuh" 
#include "libgpmlog.cuh"
#include "gpm-helper.cuh"
#include "search.cuh"

#define NTHREADS_PER_BLK 512
#define NBLKS 144
#define TIME_NOW std::chrono::high_resolution_clock::now()

__global__ void searchMemtables(const Memtable *active, const Memtable *immutable, const char* queryDataStart, const char* queryDataEnd, const uint64_t numQueries, int keyLength, int* idx_result, int* numKeysInRange, uint64_t active_size, uint64_t immutable_size) 
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(uint64_t i = tid; i < numQueries; i += NTHREADS_PER_BLK * NBLKS) {
        const char* startKey = queryDataStart + i * keyLength;
        const char* endKey = queryDataEnd + i * keyLength;
        //printf("key: %s\t", startKey); 

        // Find the index of the first key that is greater than or equal to the start key in the active memtable
        int startIndex1 = binarySearchGreater(active->d_sortedKeys, 0, active_size - 1, startKey, keyLength);
        int endIndex1 = binarySearchSmaller(active->d_sortedKeys, startIndex1, active_size - 1, endKey, keyLength);

        // Find the index of the first key that is greater than or equal to the start key in the immutable memtable
        int startIndex2 = 0; 
        int endIndex2 = 0;
        if (immutable_size != 0) {
            startIndex2 = binarySearchGreater(immutable->d_sortedKeys, 0, immutable_size - 1, startKey, keyLength);
            endIndex2 = binarySearchSmaller(immutable->d_sortedKeys, startIndex2, immutable_size - 1, endKey, keyLength);
        }

        // Store the indexes in the idx_result array
        idx_result[4 * i] = startIndex1;
        idx_result[4 * i + 1] = endIndex1;
        idx_result[4 * i + 2] = startIndex2;
        idx_result[4 * i + 3] = endIndex2;

        // Calculate the number of keys in the range for each memtable, and store the result in numKeysInRange
        int numKeys1 = endIndex1 - startIndex1 + 1;
        int numKeys2 = endIndex2 - startIndex2 + 1;
        //printf("startIndex1: %d, endIndex1: %d numKeys1: %d numKeys2: %d\t", startIndex1, endIndex1, numKeys1, numKeys2); 
        numKeysInRange[i] = numKeys1 + numKeys2;
        //__threadfence(); 

        //numKeysInRange[i] = numKeys1;
        //printf("i: %llu, keysInRange: %d, numKeysInRange: %d\t", i, keysInRange, numKeysInRange[i]);

        // Set the result pointer to the first value pointer in the active memtable
        //results[i] = active.d_sortedValuePointers[startIndex1];

    }

}

__global__ void prefixSum(const int* input, int* output, const int numElements) 
{
    __shared__ int sdata[NTHREADS_PER_BLK];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < numElements) ? input[i] : 0;
    // Reduce
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
    }

    // Write the results to the output array
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }

    for (unsigned int s = 1; s < blockDim.x; s <<= 1) {
        __syncthreads();
        if (tid < blockDim.x - s) {
            sdata[tid + s] += sdata[tid];
        }
    }

    __syncthreads();
    if (i < numElements) {
        output[i] = sdata[tid];
    }
}

__global__ void sumKernel(int* d_in, int* d_out, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ int sdata[];
    sdata[threadIdx.x] = (tid < size) ? d_in[tid] : 0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}

__global__
void memcpyKernel(char** valuePtrs, Memtable *activeMemtable, Memtable *immutableMemtable, uint64_t active_size, uint64_t immutable_size, int* indexes, int* numKeysInRange, uint64_t numQueries) 
{

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(uint64_t i = tid; i < numQueries; i += NTHREADS_PER_BLK * NBLKS) {  
        for(int j = 0; j < 100; j++) {
            valuePtrs[i * 100 + j] = activeMemtable->d_sortedValuePointers[indexes[i] + j]; 
        }
    }
}


void rangeOnGPU(char* startKeys, char* endKeys, Memtable* activeMemtable, Memtable* immutableMemtable, int keyLength, uint64_t numRange, char** &rangeValuePointers, uint64_t* rangeOperationIds)
{
    char *d_startKeys, *d_endKeys;
    cout << "numrange: " << numRange << " keyLength: " << keyLength << "\n"; 
    cudaMalloc((void**) &d_startKeys, numRange * keyLength); 
    cudaMalloc((void**) &d_endKeys, numRange * keyLength); 
    cudaError_t err = cudaPeekAtLastError();
    printf("Error %d cudaPeekerror\n", err);
    auto start = TIME_NOW; 
    cudaMemcpy(d_startKeys, startKeys, numRange * keyLength, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_endKeys, endKeys, numRange * keyLength, cudaMemcpyHostToDevice); 
    auto rangeSetupTime = (TIME_NOW - start).count(); 
    err = cudaPeekAtLastError();
    printf("Error %d cudaPeekerror\n", err);
    int *d_numKeysInRange, *d_idxResults;
    start = TIME_NOW; 
    cudaMalloc((void**) &d_numKeysInRange, numRange * sizeof(int)); 
    cudaMalloc((void**) &d_idxResults, 4 * numRange * sizeof(int)); 
    rangeSetupTime += (TIME_NOW - start).count(); 
    err = cudaPeekAtLastError();
    printf("Error %d cudaPeekerror\n", err);
    cout << "sizes: " << activeMemtable->size << " " << immutableMemtable->size; 
    start = TIME_NOW; 
    searchMemtables<<<NBLKS, NTHREADS_PER_BLK>>>(activeMemtable, immutableMemtable, d_startKeys, d_endKeys, numRange, keyLength, d_idxResults, d_numKeysInRange, activeMemtable->size, immutableMemtable->size); 
    cudaDeviceSynchronize(); 
    auto searchMemtableTime = (TIME_NOW - start).count(); 
    err = cudaPeekAtLastError();
    printf("Error %d cudaPeekerror\n", err);
    
    int blockSize = 512;
    int numBlocks = (numRange + blockSize - 1) / blockSize;

    int *d_out;

    // Assume that h_in points to your input array on the host
    cudaMallocManaged(&d_out, numBlocks * sizeof(int));
    start = TIME_NOW; 
    //sumKernel<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(d_numKeysInRange, d_out, numRange);
    prefixSum<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(d_numKeysInRange, d_out, numRange); 
    cudaDeviceSynchronize(); 
    auto sumKernelTime = (TIME_NOW - start).count(); 
    err = cudaPeekAtLastError();
    printf("Error %d cudaPeekerror\n", err);

    cout << "Total number of elements: " << d_out[0] << "\n"; 

    char** valuePtrs;

    start = TIME_NOW; 
    cudaHostAlloc(&valuePtrs, sizeof(char*) * numRange * 100, 0); 

    memcpyKernel<<<NBLKS, NTHREADS_PER_BLK>>>(valuePtrs, activeMemtable, immutableMemtable, activeMemtable->size, immutableMemtable->size, d_idxResults, d_numKeysInRange, numRange); 
    cudaDeviceSynchronize(); 
    auto memcpyTime = (TIME_NOW - start).count(); 
    
    err = cudaPeekAtLastError();
    printf("Error %d cudaPeekerror\n", err);
    
    cout << "range_setup_time: " << rangeSetupTime/1000000.0 << "\n"; 
    cout << "search_memtable_time: " << searchMemtableTime/1000000.0 << "\n"; 
    cout << "sum_kernel_time: " << sumKernelTime/1000000.0 << "\n"; 
    cout << "mempcy_kernel_time: " << memcpyTime/1000000.0 << "\n"; 

} 
