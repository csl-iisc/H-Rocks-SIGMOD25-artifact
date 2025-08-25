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
#include <cuda_runtime_api.h>

#include "libgpm.cuh"
#include "gpm-helper.cuh"
#include "batch.h"

extern "C" 
{
#include "change-ddio.h"
}

#include "libgpmlog.cuh"

#define MAX_THREADS_PER_BLOCK 1024
#define NTHREADS_PER_BLOCK 512
#define NBLOCKS 216 

#define TIME_NOW std::chrono::high_resolution_clock::now()
#define ll long long
using namespace std;

//#define __PRINT_DEBUG__

using namespace std;

typedef struct log_entry_s {
        char key[8]; 
        uint64_t sequenceId; 
        char* value; 
    } log_entry_t; 


namespace ROCKSDB_GPU 
{
    static __device__ int getGlobalIdx()
    {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;

        int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;

        return threadId;
    }

    __device__ 
        void device_strcat(char* dest, unsigned char* src, uint32_t len) 
        {
            for(uint32_t i=0; i<len; ++i) {
                //printf("i: %llu\t", i); 
                dest[i] = src[i]; 
            }
        }

    int printSortedOutput(thrust::host_vector<unsigned int> valuesSorted, thrust::host_vector<unsigned char> stringVals, int numElems, int keyLength, 
            char inputFile[500]) {
        int retval = 0;
        char outFile[500];

        sprintf(outFile,"%s_string_sort_output",inputFile);

        printf("[DEBUG] Writing Output to file %s\n", outFile);
        FILE *fp = fopen(outFile,"w");

        for(unsigned int i = 0; i < numElems; ++i) {
            //printf("New key: ");
            unsigned int index = valuesSorted[i];
            //printf("Index: %d ", index);
            while(true) { 
                char ch;
                ch = (char)(stringVals[index]);
                if(ch == '\0') break;
                fprintf(fp,"%c",ch);
                //printf("%c", ch); 
                index++;
            }
            fprintf(fp,"\n");
            //printf("\n");
        }	
        return retval;
    }

    double calculateDiff (struct timespec t1, struct timespec t2) { 
        return (((t1.tv_sec - t2.tv_sec)*1000.0) + (((t1.tv_nsec - t2.tv_nsec)*1.0)/1000000.0));
    }

    __global__ void findSuccessor( unsigned char *d_array_stringVals, unsigned long long int *d_array_segment_keys,  
            unsigned int *d_array_valIndex, unsigned long long int *d_array_segment_keys_out,  unsigned int numElems, 
            unsigned int keyLength, unsigned int charPosition, unsigned int segmentBytes) {

        int threadID = (blockIdx.x * blockDim.x) +  threadIdx.x;
        if(threadID > numElems) return;
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
                if( stringIndex +  startPosition < keyLength ) { 
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
            unsigned int *d_array_segment, unsigned int segmentBytes, unsigned int numElems) { 

        int threadID = (blockIdx.x * blockDim.x) +  threadIdx.x;
        if(threadID >= numElems) return;

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

    void sort(char* putKeys, uint64_t numElems, int keyLength, Memtable &table)
    {


        thrust::host_vector<unsigned long long int> h_keys(numElems);
        // This should be initialized with the index of Put in the batch 
        thrust::host_vector<unsigned int> h_valIndex(numElems);
        // This is the putKeys array 
        thrust::host_vector<unsigned char> h_stringVals(putKeys, putKeys + numElems); 
        printf("keys: %s\n", putKeys);

        // Can later pass the file name 

        int nthreads = 128; 
        uint64_t numKeysPerThread = numElems/nthreads; 
        auto start = TIME_NOW;
#pragma omp parallel for num_threads(nthreads)
        for(uint64_t i = 0; i < numElems; ++i) {
            h_valIndex[i] = i * keyLength; 
            unsigned long long int firstKey = 0;
            for (unsigned int prefixLen = 0; prefixLen < 8; prefixLen++) {
                unsigned char ch = (unsigned char) putKeys[i * keyLength + prefixLen]; 
                firstKey |= (((unsigned long long int) ch) << ((7 - prefixLen) * 8)); 

            }
            h_keys[i] = firstKey;
        }

        thrust::device_vector<unsigned char> d_stringVals = h_stringVals;
        thrust::device_vector<unsigned long long int> d_segment_keys = h_keys;
        thrust::device_vector<unsigned int> d_valIndex = h_valIndex;
        thrust::device_vector<unsigned int> d_static_index(numElems);
        thrust::device_vector<unsigned int> d_output_valIndex(numElems);

        thrust::sequence(d_static_index.begin(), d_static_index.begin() + numElems);

        auto setup_time = (TIME_NOW - start).count(); 
        cudaError_t err = cudaPeekAtLastError();

        unsigned int charPosition = 8;
        unsigned int originalSize = numElems;
        unsigned int segmentBytes = 0;
        unsigned int lastSegmentID = 0;

        unsigned int numSorts = 0;
        unsigned char* d_array_stringVals; 
        unsigned int* d_array_valIndex; 


        start = TIME_NOW; 
        while(true) { 

            thrust::sort_by_key (
                    d_segment_keys.begin(),
                    d_segment_keys.begin() + numElems,
                    d_valIndex.begin()
                    ); 
            numSorts++;

            thrust::device_vector<unsigned long long int> d_segment_keys_out(numElems, 0);

            d_array_stringVals = thrust::raw_pointer_cast(&d_stringVals[0]); 
            d_array_valIndex = thrust::raw_pointer_cast(&d_valIndex[0]);
            unsigned int *d_array_static_index = thrust::raw_pointer_cast(&d_static_index[0]);
            unsigned int *d_array_output_valIndex = thrust::raw_pointer_cast(&d_output_valIndex[0]);

            unsigned long long int *d_array_segment_keys_out = thrust::raw_pointer_cast(&d_segment_keys_out[0]);
            unsigned long long int *d_array_segment_keys = thrust::raw_pointer_cast(&d_segment_keys[0]); 

            int numBlocks = 1;
            int numThreadsPerBlock = numElems/numBlocks;

            if(numThreadsPerBlock > MAX_THREADS_PER_BLOCK) { 
                numBlocks = (int)ceil(numThreadsPerBlock/(float)MAX_THREADS_PER_BLOCK);
                numThreadsPerBlock = MAX_THREADS_PER_BLOCK;
            }
            dim3 grid(numBlocks, 1, 1);
            dim3 threads(numThreadsPerBlock, 1, 1); 

            cudaDeviceSynchronize();
            std::cout << "Grid: " << numBlocks << " threads: " << numThreadsPerBlock << std::endl; 
            findSuccessor<<<grid, threads, 0>>>(d_array_stringVals, d_array_segment_keys, d_array_valIndex, 
                    d_array_segment_keys_out, numElems, keyLength, charPosition, segmentBytes);
            cudaDeviceSynchronize();
            cudaError_t err = cudaPeekAtLastError();
            printf("Error %d cudaPeekerror\n", err);
            std::cout << cudaGetErrorName (err) << std::endl; 

            charPosition+=7;

            thrust::device_vector<unsigned int> d_temp_vector(numElems);
            thrust::device_vector<unsigned int> d_segment(numElems);
            thrust::device_vector<unsigned int> d_stencil(numElems);
            thrust::device_vector<unsigned int> d_map(numElems);

            unsigned int *d_array_temp_vector = thrust::raw_pointer_cast(&d_temp_vector[0]);
            unsigned int *d_array_segment = thrust::raw_pointer_cast(&d_segment[0]);
            unsigned int *d_array_stencil = thrust::raw_pointer_cast(&d_stencil[0]);


            thrust::transform(d_segment_keys_out.begin(), d_segment_keys_out.begin() + numElems, d_temp_vector.begin(), get_segment_bytes());

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

            thrust::inclusive_scan(d_temp_vector.begin(), d_temp_vector.begin() + numElems, d_segment.begin());

            cudaDeviceSynchronize(); 
            eliminateSingleton<<<grid, threads, 0>>>(d_array_output_valIndex, d_array_valIndex, d_array_static_index, 
                    d_array_temp_vector, d_array_stencil, numElems); 
            cudaDeviceSynchronize();

#ifdef __PRINT_DEUBG__
            cout << "Stencil values are ";
            for( itr = d_stencil.begin(); itr != d_stencil.end(); ++itr) { 
                cout << *itr << " ";
            }
            cout << endl;
#endif

            thrust::exclusive_scan(d_stencil.begin(), d_stencil.begin() + numElems, d_map.begin());

            thrust::scatter_if(d_segment.begin(), d_segment.begin() + numElems, d_map.begin(), 
                    d_stencil.begin(), d_temp_vector.begin());
            thrust::copy(d_temp_vector.begin(), d_temp_vector.begin() + numElems, d_segment.begin()); 

            thrust::scatter_if(d_valIndex.begin(), d_valIndex.begin() + numElems, d_map.begin(), 
                    d_stencil.begin(), d_temp_vector.begin());
            thrust::copy(d_temp_vector.begin(), d_temp_vector.begin() + numElems, d_valIndex.begin()); 

            thrust::scatter_if(d_static_index.begin(), d_static_index.begin() + numElems, d_map.begin(), 
                    d_stencil.begin(), d_temp_vector.begin());
            thrust::copy(d_temp_vector.begin(), d_temp_vector.begin() + numElems, d_static_index.begin()); 

            thrust::scatter_if(d_segment_keys_out.begin(), d_segment_keys_out.begin() + numElems, d_map.begin(), 
                    d_stencil.begin(), d_segment_keys.begin());
            thrust::copy(d_segment_keys.begin(), d_segment_keys.begin() + numElems, d_segment_keys_out.begin()); 


            numElems = *(d_map.begin() + numElems - 1) + *(d_stencil.begin() + numElems - 1); 
            if(numElems != 0) { 
                lastSegmentID = *(d_segment.begin() + numElems - 1);
            }

            d_temp_vector.clear();
            d_temp_vector.shrink_to_fit();

            d_stencil.clear();
            d_stencil.shrink_to_fit();

            d_map.clear();
            d_map.shrink_to_fit();

            if(numElems == 0) {
                thrust::copy(d_output_valIndex.begin(), d_output_valIndex.begin() + originalSize, h_valIndex.begin());
                break;
            }

            segmentBytes = (int) ceil(((float)(log2((float)lastSegmentID+2))*1.0)/8.0);
            cout << "segmentBytes: " << segmentBytes << "\n"; 
            charPosition-=(segmentBytes-1);

#ifdef __PRINT_DEBUG__
            printf("[DEBUG] numElems %d, charPosition %d, lastSegmentID %d, segmentBytes %d\n", numElems, 
                    charPosition, lastSegmentID, segmentBytes );
#endif

            int numBlocks1 = 1;
            int numThreadsPerBlock1 = numElems/numBlocks1;

            if(numThreadsPerBlock1 > MAX_THREADS_PER_BLOCK) { 
                numBlocks1 = (int)ceil(numThreadsPerBlock1/(float)MAX_THREADS_PER_BLOCK);
                numThreadsPerBlock1 = MAX_THREADS_PER_BLOCK;
            }
            dim3 grid1(numBlocks1, 1, 1);
            dim3 threads1(numThreadsPerBlock1, 1, 1); 

            cudaDeviceSynchronize();
            rearrangeSegMCU<<<grid1, threads1, 0>>>(d_array_segment_keys, d_array_segment_keys_out, d_array_segment, 
                    segmentBytes, numElems);
            cudaDeviceSynchronize();

#ifdef __PRINT_DEBUG__		
            printf("---------- new keys are --------\n");
            itr2 = d_segment_keys.begin();
            unsigned int ct = 0;
            for( ct = 0; ct < numElems; ct++ ) { 
                print_chars(*itr2, segmentBytes);
                printf("\n");
                ++itr2;
            }
            printf("----\n");
#endif
        }
        auto sort_time = (TIME_NOW - start).count();

        thrust::device_vector<unsigned int> d_copy = h_valIndex; 
        unsigned int *d_sorted_index; 
        d_sorted_index = thrust::raw_pointer_cast(&d_copy[0]);


        printf("setup_time (in ms): %f\n", setup_time/1000000.0); 
        printf("sort_time (in ms): %f\n", sort_time/1000000.0); 

    }
}
