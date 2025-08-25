#pragma once

#define KEY_LEN 8 
#define NUM_WAYS 4 
#define CACHE_SIZE 10000000

struct CacheLine {
    public:
        long long int tag;
        char key[KEY_LEN];
        char* value;
        unsigned int frequency;
        bool invalidated;
};

struct CacheSet {
    public:
        CacheLine lines[NUM_WAYS];
        pthread_mutex_t lock;
};

struct BCache {
    public:
        CacheSet sets[CACHE_SIZE];
};

//unsigned long long hash_fnv1a(char* key);
class BlockCache {
    public:
    void put(BCache* cache, char* key, char* value);
    char* get(BCache* cache, char* key);
    void invalidate(BCache* cache, char* key);
    //__host__ __device__ void cacheReadKernel(char** keys, int numKeys, char** values, int keyLength);

    //static const int CACHE_SIZE = 10000000;
    //static const int NUM_WAYS = 4;
    //static const int KEY_LEN = 8;

    //BCache* cache; 
    //__host__ __device__ unsigned long long hash_fnv1a_device(char* key, int length);
    //__device__ bool string_cmp(const char* a, const char* b, size_t length);
    BCache* createCache(BCache *cache);
    void freeCache(BCache *cache);
};

