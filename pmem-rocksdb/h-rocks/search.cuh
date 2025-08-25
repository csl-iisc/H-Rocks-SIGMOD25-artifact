#pragma once 

__device__ int binarySearchGreater(const char* sortedKeys, int start, int end, const char* key, int keyLength); 
__device__ int binarySearchSmaller(const char* sortedKeys, int start, int end, const char* key, int keyLength); 
__device__ bool binarySearch(const char* sortedKeys, uint64_t start, uint64_t end, const char* key, uint64_t& index, int keyLength); 
