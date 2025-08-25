#include <iostream>
#include <cstdint>
#include <cstring>

uint64_t fnv1a_hash(const char* data, int length) 
{
    const uint64_t FNV_offset_basis = 14695981039346656037ULL;
    const uint64_t FNV_prime = 1099511628211ULL;

    uint64_t hash = FNV_offset_basis;

    for (int i = 0; i < length; i++) {
        hash ^= static_cast<uint64_t>(data[i]);
        hash *= FNV_prime;

    }

    return hash;

}

__device__ uint64_t fnv1a_hash_gpu(const char* data, int length) 
{
    const uint64_t FNV_offset_basis = 14695981039346656037ULL;
    const uint64_t FNV_prime = 1099511628211ULL;

    uint64_t hash = FNV_offset_basis;

    for (int i = 0; i < length; i++) {
        hash ^= static_cast<uint64_t>(data[i]);
        hash *= FNV_prime;
    }
    return hash;
}
