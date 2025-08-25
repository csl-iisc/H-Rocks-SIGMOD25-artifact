#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <sys/mman.h>
#include <fcntl.h>
#include "utree.h"
#include <gperftools/profiler.h>
#include <unistd.h>
using namespace std;
typedef uint64_t setkey_t;
typedef void *setval_t;

#define OP_NUM 10000000
setkey_t keys[OP_NUM];

struct timeval start_time, end_time;
uint64_t time_interval;

char * start_addr;
char * curr_addr;

int main(int argc, char **argv)
{
    int fd = open("/pmem", O_RDWR);
    void *pmem = mmap(NULL, (uint64_t)6 * 1024ULL * 1024ULL * 1024ULL,
                      PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    start_addr = (char *)pmem;
    curr_addr = start_addr;
    printf("start_addr=%p, end_addr=%p\n", start_addr,
           start_addr + (uint64_t)6 * 1024ULL * 1024ULL * 1024ULL);
    btree *bt;
    bt = new btree();
	
    int option_char, nthreads, key_size, value_size;
     uint64_t num_puts;
     std::string file_loc;

     while ((option_char = getopt (argc, argv, ":n:k:v:t:")) != -1){
         switch (option_char)
         {
             case 'n': num_puts = atoi (optarg); break;
             case 'k': key_size = atoi (optarg); break;
             case 'v': value_size = atoi (optarg); break;
             case 't': nthreads = atoi (optarg); break;
             case ':': fprintf (stderr, "option needs a value\n");
             case '?': fprintf (stderr, "usage: %s [-n <number of keys>] [-k <key size>] [-v <value  size>]\n", argv[0]);
         }
     }
     std::cout << "Number of put keys: " << num_puts << std::endl;

    int key = 1000, value = 1000; 
    for(uint64_t i = 0; i < num_puts; ++i) {
        value++; 
        bt->insert(key, (char*)value); 
        char* obtained_value = bt->search(key); 
        std::cout << "obtained value: " << obtained_value << " value: " << value << "\n"; 
    }


}
