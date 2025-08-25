#pragma once 

uint64_t fnv1a_hash(const char* data, int length);
__device__ uint64_t fnv1a_hash_gpu(const char* data, int length); 
