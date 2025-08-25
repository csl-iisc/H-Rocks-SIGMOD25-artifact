#include <cuda_runtime_api.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <cstdlib>
#include <math.h>

#include <cstdio>
#include <string>
#include <fstream>
#include <iostream>
#include <getopt.h>
#include <unistd.h>
#include "command.h"
#include "memtable.cuh"

double calculateDiff (struct timespec t1, struct timespec t2) { 
    return (((t1.tv_sec - t2.tv_sec)*1000.0) + (((t1.tv_nsec - t2.tv_nsec)*1.0)/1000000.0));
}

__global__ void findSuccessor( unsigned char *d_array_stringVals, unsigned long long int *d_array_segment_keys,  
        unsigned int *d_array_valIndex, unsigned long long int *d_array_segment_keys_out,  unsigned int numElements, 
        unsigned int stringSize, unsigned int charPosition, unsigned int segmentBytes) {

    int threadID = (blockIdx.x * blockDim.x) +  threadIdx.x;
    if(threadID > numElements) return;
    d_array_segment_keys_out[threadID] = 0;

    if(threadID > 0) { 
        if(d_array_segment_keys[threadID] != d_array_segment_keys[threadID-1]) { 
            d_array_segment_keys_out[threadID] = ((unsigned long long int)(1) << 56);
        }
    }

    unsigned int stringIndex = d_array_valIndex[threadID];
    unsigned long long int currentKey = (d_array_segment_keys[threadID] << (segmentBytes*8));
    unsigned char ch;
    int i = 0;
    unsigned int end = 0;

    for(i = 7; i >= ((int)segmentBytes); i--) { 
        ch = (unsigned char)(currentKey >> (i*8));
        if(ch == '\0') { 
            end = 1;
            break;
        }
    }

    if( end == 0) {
        unsigned int startPosition = charPosition;
        for(i = 6; i >=0; i--) { 
            if( stringIndex +  startPosition < stringSize ) { 
                ch = d_array_stringVals[ stringIndex + startPosition ];
                d_array_segment_keys_out[threadID] |= ((unsigned long long int) ch << (i*8)); 
                startPosition++;
                if(ch == '\0') break;
            }
            if(ch == '\0') break;
        }

    } else { 
        d_array_segment_keys_out[threadID] = ((unsigned long long int)(1) << 56);
    }
}

__global__ void  eliminateSingleton(unsigned int *d_array_output_valIndex, unsigned int *d_array_valIndex, unsigned int *d_array_static_index, 
        unsigned int *d_array_map, unsigned int *d_array_stencil, int currentSize) {

    int threadID = (blockIdx.x * blockDim.x) +  threadIdx.x;
    if(threadID >= currentSize) return;

    d_array_stencil[threadID] = 1;

    if(threadID == 0 && (d_array_map[threadID + 1] == 1)) { 
        d_array_stencil[threadID] = 0; 
    } else if( (threadID == (currentSize-1)) && (d_array_map[threadID] == 1) ) {
        d_array_stencil[threadID] = 0;  
    } else if( (d_array_map[threadID] == 1) && (d_array_map[threadID + 1] == 1)) { 
        d_array_stencil[threadID] = 0; 
    }

    if(d_array_stencil[threadID] == 0) { 
        d_array_output_valIndex[ d_array_static_index[threadID] ] = d_array_valIndex[threadID]; 
    }
}

__global__ void rearrangeSegMCU(unsigned long long int *d_array_segment_keys, unsigned long long int *d_array_segment_keys_out, 
        unsigned int *d_array_segment, unsigned int segmentBytes, unsigned int numElements) { 

    int threadID = (blockIdx.x * blockDim.x) +  threadIdx.x;
    if(threadID >= numElements) return;

    unsigned long long int currentKey = (d_array_segment_keys_out[threadID] << 8);
    unsigned long long int segmentID  = (unsigned long long int) d_array_segment[threadID];
    d_array_segment_keys[threadID] = (segmentID << ((8-segmentBytes)*8));
    d_array_segment_keys[threadID] |= (currentKey >> (segmentBytes*8));
    return;
}

struct get_segment_bytes {
    __host__ __device__
        unsigned int operator()(const unsigned long long int& x) const { 
            return (unsigned int)(x >> 56);
        }
};

void print_chars(unsigned long long int val, unsigned int segmentBytes) { 
    printf("printing keys \t");
    int shift = 56;
    if(segmentBytes > 0) { 
        printf("segment number %d \t", (unsigned int)(val>>((8-segmentBytes)*8)));
        shift-=(segmentBytes*8);
    }
    while(shift>=0) {
        char ch = (char)(val>> shift);
        printf("%c", ch);
        shift-=8;
        if(ch == '\0') printf("*");
    }
    printf(" ");
}


void initialize_buckets(unsigned char* inbuf, int keyLen, int numKeys, thrust::host_vector<unsigned char> h_stringVals, thrust::host_vector<unsigned int> h_valIndex, thrust::host_vector<unsigned long long int> h_keys) 
{
    // Copy inbuf to host_vector
    //h_stringVals.push_back(inbuf); 
    for(int i=0; i<numKeys; ++i) {
        cout << i << " "; 
        h_valIndex[i] = i * keyLen; 
        h_stringVals[(i+1) * keyLen] = '\0'; 
        unsigned int prefixLen = 0;
        unsigned long long int firstKey = 0;
        for (int j = 0; j<min(keyLen,8); j++) {
            firstKey |= (((unsigned long long int) h_stringVals[i * keyLen + j] << (7 - prefixLen) * 8)); 
            prefixLen++; 
        }
        h_keys[i] = firstKey; 
    }
}


void read_file(char *filename, unsigned char *inbuf, int numKeys, int keyLen)
{
    ifstream in(filename); 
    int i=0; 
    if (in.is_open()) {
        while ( !in.eof()) {
            in >> inbuf[i]; 
            i++; 
        }
    }
    in.close(); 
}

void sort(uint64_t numKeys, char* keys, char** values, uint32_t keyLen, uint32_t valueLen, uint64_t keySize, uint64_t valueSize, char* sortedKeys, char** sortedValues, uint64_t* sequences, uint64_t* sortedSeq, bool* filter, BatchElement* batchQ) 
{
    uint64_t numElements = numKeys; 
    uint32_t stringSize = keySize;

    thrust::host_vector<unsigned long long int> h_keys(numKeys);
    // This should be initialized with the index of Put in the batch 
    thrust::host_vector<unsigned int> h_valIndex(numKeys);
    // This is the putKeys array 
    thrust::host_vector<unsigned char> h_stringVals(keys, keys + keySize); 
    printf("keys: %s\n", keys);

    // Can later pass the file name 

    int nthreads = 128; 
    uint64_t numKeysPerThread = numKeys/nthreads; 
    auto start = TIME_NOW;
#pragma omp parallel for num_threads(nthreads)
    for(uint64_t i = 0; i < numKeys; ++i) {
        h_valIndex[i] = i * keyLen; 
        unsigned long long int firstKey = 0;
        for (unsigned int prefixLen = 0; prefixLen < 8; prefixLen++) {
            unsigned char ch = (unsigned char) keys[i * keyLen + prefixLen]; 
            firstKey |= (((unsigned long long int) ch) << ((7 - prefixLen) * 8)); 

        }
        h_keys[i] = firstKey;
    }

    thrust::device_vector<unsigned char> d_stringVals = h_stringVals;
    thrust::device_vector<unsigned long long int> d_segment_keys = h_keys;
    thrust::device_vector<unsigned int> d_valIndex = h_valIndex;
    thrust::device_vector<unsigned int> d_static_index(numElements);
    thrust::device_vector<unsigned int> d_output_valIndex(numElements);

    thrust::sequence(d_static_index.begin(), d_static_index.begin() + numElements);

    auto setup_time = (TIME_NOW - start).count(); 
    cudaError_t err = cudaPeekAtLastError();

    unsigned int charPosition = 8;
    unsigned int originalSize = numElements;
    unsigned int segmentBytes = 0;
    unsigned int lastSegmentID = 0;

    unsigned int numSorts = 0;
    unsigned char* d_array_stringVals; 
    unsigned int* d_array_valIndex; 


    start = TIME_NOW; 
    while(true) { 

        thrust::sort_by_key (
                d_segment_keys.begin(),
                d_segment_keys.begin() + numElements,
                d_valIndex.begin()
                ); 
        numSorts++;

        thrust::device_vector<unsigned long long int> d_segment_keys_out(numElements, 0);

        d_array_stringVals = thrust::raw_pointer_cast(&d_stringVals[0]); 
        d_array_valIndex = thrust::raw_pointer_cast(&d_valIndex[0]);
        unsigned int *d_array_static_index = thrust::raw_pointer_cast(&d_static_index[0]);
        unsigned int *d_array_output_valIndex = thrust::raw_pointer_cast(&d_output_valIndex[0]);

        unsigned long long int *d_array_segment_keys_out = thrust::raw_pointer_cast(&d_segment_keys_out[0]);
        unsigned long long int *d_array_segment_keys = thrust::raw_pointer_cast(&d_segment_keys[0]); 

        int numBlocks = 1;
        int numThreadsPerBlock = numElements/numBlocks;

        if(numThreadsPerBlock > MAX_THREADS_PER_BLOCK) { 
            numBlocks = (int)ceil(numThreadsPerBlock/(float)MAX_THREADS_PER_BLOCK);
            numThreadsPerBlock = MAX_THREADS_PER_BLOCK;
        }
        dim3 grid(numBlocks, 1, 1);
        dim3 threads(numThreadsPerBlock, 1, 1); 

        cudaDeviceSynchronize();
        std::cout << "Grid: " << numBlocks << " threads: " << numThreadsPerBlock << std::endl; 
        findSuccessor<<<grid, threads, 0>>>(d_array_stringVals, d_array_segment_keys, d_array_valIndex, 
                d_array_segment_keys_out, numElements, stringSize, charPosition, segmentBytes);
        cudaDeviceSynchronize();
        cudaError_t err = cudaPeekAtLastError();
        printf("Error %d cudaPeekerror\n", err);
        std::cout << cudaGetErrorName (err) << std::endl; 

        charPosition+=7;

        thrust::device_vector<unsigned int> d_temp_vector(numElements);
        thrust::device_vector<unsigned int> d_segment(numElements);
        thrust::device_vector<unsigned int> d_stencil(numElements);
        thrust::device_vector<unsigned int> d_map(numElements);

        unsigned int *d_array_temp_vector = thrust::raw_pointer_cast(&d_temp_vector[0]);
        unsigned int *d_array_segment = thrust::raw_pointer_cast(&d_segment[0]);
        unsigned int *d_array_stencil = thrust::raw_pointer_cast(&d_stencil[0]);


        thrust::transform(d_segment_keys_out.begin(), d_segment_keys_out.begin() + numElements, d_temp_vector.begin(), get_segment_bytes());

#ifdef __PRINT_DEBUG__
        thrust::device_vector<unsigned int>::iterator itr;
        thrust::device_vector<unsigned long long int>::iterator itr2;
        thrust::device_vector<unsigned long long int>::iterator itr3;


        itr2 = d_segment_keys_out.begin();
        itr3 = d_segment_keys.begin();

        for(itr = d_temp_vector.begin(); itr!=d_temp_vector.end(); ++itr) { 
            cout << *itr << " ";
            print_chars(*itr3, segmentBytes);
            cout << " ";
            print_chars(*itr2, 1);
            ++itr2;
            ++itr3;
            cout << endl;
        }
#endif

        thrust::inclusive_scan(d_temp_vector.begin(), d_temp_vector.begin() + numElements, d_segment.begin());

        cudaDeviceSynchronize(); 
        eliminateSingleton<<<grid, threads, 0>>>(d_array_output_valIndex, d_array_valIndex, d_array_static_index, 
                d_array_temp_vector, d_array_stencil, numElements); 
        cudaDeviceSynchronize();

#ifdef __PRINT_DEUBG__
        cout << "Stencil values are ";
        for( itr = d_stencil.begin(); itr != d_stencil.end(); ++itr) { 
            cout << *itr << " ";
        }
        cout << endl;
#endif

        thrust::exclusive_scan(d_stencil.begin(), d_stencil.begin() + numElements, d_map.begin());

        thrust::scatter_if(d_segment.begin(), d_segment.begin() + numElements, d_map.begin(), 
                d_stencil.begin(), d_temp_vector.begin());
        thrust::copy(d_temp_vector.begin(), d_temp_vector.begin() + numElements, d_segment.begin()); 

        thrust::scatter_if(d_valIndex.begin(), d_valIndex.begin() + numElements, d_map.begin(), 
                d_stencil.begin(), d_temp_vector.begin());
        thrust::copy(d_temp_vector.begin(), d_temp_vector.begin() + numElements, d_valIndex.begin()); 

        thrust::scatter_if(d_static_index.begin(), d_static_index.begin() + numElements, d_map.begin(), 
                d_stencil.begin(), d_temp_vector.begin());
        thrust::copy(d_temp_vector.begin(), d_temp_vector.begin() + numElements, d_static_index.begin()); 

        thrust::scatter_if(d_segment_keys_out.begin(), d_segment_keys_out.begin() + numElements, d_map.begin(), 
                d_stencil.begin(), d_segment_keys.begin());
        thrust::copy(d_segment_keys.begin(), d_segment_keys.begin() + numElements, d_segment_keys_out.begin()); 


        numElements = *(d_map.begin() + numElements - 1) + *(d_stencil.begin() + numElements - 1); 
        if(numElements != 0) { 
            lastSegmentID = *(d_segment.begin() + numElements - 1);
        }

        d_temp_vector.clear();
        d_temp_vector.shrink_to_fit();

        d_stencil.clear();
        d_stencil.shrink_to_fit();

        d_map.clear();
        d_map.shrink_to_fit();

        if(numElements == 0) {
            thrust::copy(d_output_valIndex.begin(), d_output_valIndex.begin() + originalSize, h_valIndex.begin());
            break;
        }

        segmentBytes = (int) ceil(((float)(log2((float)lastSegmentID+2))*1.0)/8.0);
        cout << "segmentBytes: " << segmentBytes << "\n"; 
        charPosition-=(segmentBytes-1);

#ifdef __PRINT_DEBUG__
        printf("[DEBUG] numElements %d, charPosition %d, lastSegmentID %d, segmentBytes %d\n", numElements, 
                charPosition, lastSegmentID, segmentBytes );
#endif

        int numBlocks1 = 1;
        int numThreadsPerBlock1 = numElements/numBlocks1;

        if(numThreadsPerBlock1 > MAX_THREADS_PER_BLOCK) { 
            numBlocks1 = (int)ceil(numThreadsPerBlock1/(float)MAX_THREADS_PER_BLOCK);
            numThreadsPerBlock1 = MAX_THREADS_PER_BLOCK;
        }
        dim3 grid1(numBlocks1, 1, 1);
        dim3 threads1(numThreadsPerBlock1, 1, 1); 

        cudaDeviceSynchronize();
        rearrangeSegMCU<<<grid1, threads1, 0>>>(d_array_segment_keys, d_array_segment_keys_out, d_array_segment, 
                segmentBytes, numElements);
        cudaDeviceSynchronize();

#ifdef __PRINT_DEBUG__		
        printf("---------- new keys are --------\n");
        itr2 = d_segment_keys.begin();
        unsigned int ct = 0;
        for( ct = 0; ct < numElements; ct++ ) { 
            print_chars(*itr2, segmentBytes);
            printf("\n");
            ++itr2;
        }
        printf("----\n");
#endif
    }
    auto sort_time = (TIME_NOW - start).count();


