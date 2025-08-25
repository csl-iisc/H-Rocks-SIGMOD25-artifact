#include <iostream>
#include <vector>
#include <string>
#include "batch.h"
#include "memtable.cuh"
#include "gpu_gets.cuh"

#define NTHREADS_PER_BLK 512
#define NBLKS 144

__host__ __device__
int stringCmp(const char* a, const char* b, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

__host__ __device__
int string_cmp(const char* a, const char* b, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

__device__ bool binarySearch(const char* sortedKeys, uint64_t start, uint64_t end, const char* key, uint64_t& index, int keyLength) 
{
    while (start <= end) {
        uint64_t mid = start + (end - start) / 2;
        int cmp = string_cmp(key, sortedKeys + mid * keyLength, keyLength);
        if (cmp == 0) {
            index = mid;
            return true;
        }
        else if (cmp < 0) {
            end = mid - 1;
        }
        else {
            start = mid + 1;
        }
    }
    index = start;
    return true;
}


/*
   __device__ bool binarySearchGreater(const char* sortedKeys, int start, int end, const char* key, int& index, int keyLength) {
   while (start <= end) {
   int mid = start + (end - start) / 2;
   int cmp = stringCmp(key, sortedKeys + mid * keyLength, keyLength);
   if (cmp == 0) {
// If we find the key, return its index
index = mid;
return true;
}
else if (cmp < 0) {
end = mid - 1;
}
else {
start = mid + 1;
}
}
// If we haven't found the key, return the index of the next element
// after the last element we compared to
//index = (sortedKeys[start] < key) ? (start + 1) : start;
const char* startKey = sortedKeys + start * keyLength;
index = (stringCmp(startKey, key, keyLength) > 0) ? start : (start + 1);
return false;
}
 */

__device__ int binarySearchGreater(const char* sortedKeys, int start, int end, const char* key, int keyLength) 
{
    while (start <= end) {
        int mid = start + (end - start) / 2;
        const char* midKey = sortedKeys + mid * keyLength;
        int cmp = stringCmp(key, midKey, keyLength);
        if (cmp == 0) {
            // If we find the key, return its index
            return mid;
        }
        else if (cmp < 0) {
            end = mid - 1;
        }
        else {
            start = mid + 1;
        }
    }
    // If we haven't found the key, return the index of the next element
    // after the last element we compared to
    return (start < end) ? start : end + 1;
}


__device__ int binarySearchSmaller(const char* sortedKeys, int start, int end, const char* key, int keyLength) 
{
    while (start <= end) {
        int mid = start + (end - start) / 2;
        const char* midKey = sortedKeys + mid * keyLength;
        int cmp = stringCmp(key, midKey, keyLength);
        if (cmp <= 0) {
            // If we find the key or the key is smaller than midKey,
            // return the index of the previous element
            return (mid == 0) ? -1 : (mid - 1);
        }
        else {
            start = mid + 1;
        }
    }
    // If we haven't found the key, return the index of the last element
    // we compared to
    return end;
}


