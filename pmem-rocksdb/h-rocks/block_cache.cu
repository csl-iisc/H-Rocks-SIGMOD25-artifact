#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "block_cache.cuh"

const unsigned long long FNV_offset_basis = 14695981039346656037U;
const unsigned long long FNV_prime = 1099511628211U;

unsigned long long hash_fnv1a(char* key) {
    unsigned long long hash = FNV_offset_basis;
    size_t length = strlen(key);
    for (size_t i = 0; i < length; i++) {
        hash ^= (unsigned char)key[i];
        hash *= FNV_prime;
    }
    return hash;
}

BCache* BlockCache::createCache(BCache* cache) 
{
    cache = (BCache*) malloc(sizeof(BCache));
    cudaHostRegister(cache, sizeof(BCache), 0);
#ifdef __MANAGED__ 
    BCache* cache; 
    cudaMallocManaged(&cache, sizeof(BCache));
#endif
    cudaError_t err = cudaPeekAtLastError(); 
    printf("Error %d cudaPeekError\n", err); 

    for (int i = 0; i < CACHE_SIZE; i++) {
        for (int j = 0; j < NUM_WAYS; j++) {
            cache->sets[i].lines[j].tag = -1;
            cache->sets[i].lines[j].value = 0;
            cache->sets[i].lines[j].frequency = 0;
        }
        pthread_mutex_init(&cache->sets[i].lock, NULL);
    }
    return cache;
}

void BlockCache::put(BCache* cache, char* key, char* value) {
    unsigned long long hashed_key = hash_fnv1a(key);
    int setIndex = hashed_key % CACHE_SIZE;
    CacheSet* set = &cache->sets[setIndex];

    pthread_mutex_lock(&set->lock);

    // Find the least frequently used line in the set
    int minFrequency = set->lines[0].frequency;
    int minIndex = 0;
    bool found = false; 
    for(int i = 1; i < NUM_WAYS; i++) {
        if (set->lines[i].invalidated == true) {
            minIndex = i; 
            found = true;
        }
    }
    if (!found) {
        for (int i = 1; i < NUM_WAYS; i++) {
            if (set->lines[i].frequency < minFrequency) {
                minFrequency = set->lines[i].frequency;
                minIndex = i;
            }
        }
    }
    // Replace the least frequently used line with the new value
    set->lines[minIndex].tag = hashed_key / CACHE_SIZE;
    //set->lines[minIndex].key = strdup(key);
    strcpy(set->lines[minIndex].key, key); 
    set->lines[minIndex].value = strdup(value);
    set->lines[minIndex].frequency = 1;

    pthread_mutex_unlock(&set->lock);
}


void BlockCache::invalidate(BCache* cache, char* key) 
{
    unsigned long long hashed_key = hash_fnv1a(key);
    int setIndex = hashed_key % CACHE_SIZE;
    CacheSet* set = &cache->sets[setIndex];

    // Find the line with the matching tag
    for (int i = 0; i < NUM_WAYS; i++) {
        if (set->lines[i].tag == hashed_key / CACHE_SIZE && !strcmp(set->lines[i].key, key)) {
            set->lines[i].invalidated = true; 
        }
    }

}

char* BlockCache::get(BCache* cache, char* key) 
{
    unsigned long long hashed_key = hash_fnv1a(key);
    int setIndex = hashed_key % CACHE_SIZE;
    CacheSet* set = &cache->sets[setIndex];

    pthread_mutex_lock(&set->lock);

    // Find the line with the matching tag
    for (int i = 0; i < NUM_WAYS; i++) {
        if (set->lines[i].tag == hashed_key / CACHE_SIZE && !strcmp(set->lines[i].key, key)) {
            set->lines[i].frequency++;
            char* value = strdup(set->lines[i].value);

            pthread_mutex_unlock(&set->lock);
            return value;
        }
    }

    pthread_mutex_unlock(&set->lock);

    // If the tag was not found, return NULL
    return NULL;
}

void BlockCache::freeCache(BCache* cache) {
    for (int i = 0; i < CACHE_SIZE; i++) {
        pthread_mutex_destroy(&cache->sets[i].lock);
    }
    cudaDeviceSynchronize();

    cudaFree(cache);
}

