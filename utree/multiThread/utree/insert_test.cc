#include "utree.h"
#include "zipfian.h"
#include "zipfian_util.h"
#include <cmath>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <limits.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/time.h>

extern "C"
{
    #include <atomic_ops.h>
}  

__thread char *start_addr; 
__thread char *curr_addr; 

#define TIME_NOW std::chrono::high_resolution_clock::now()

int main(int argc, char **argv) 
{
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
            case '?': fprintf (stderr, "usage: %s [-n <number of keys>] [-k <key size>] [-v <value size>]\n", argv[0]);
        }
    }
    std::cout << "Number of put keys: " << num_puts << std::endl;

    int fd[2]; 
    void *pmem[2]; 
    uint64_t allocate_size = 20 * 1024LL * 1024LL * 1024LL; 
    fd[0] = open("/pmem", O_RDWR); 
    fd[1] = open("/pmem", O_RDWR); 
 
    for (int i=0; i<2; i++){
      pmem[i] = mmap(NULL, allocate_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd[i], 0);
      //thread_space_start_addr[i] = (char *)pmem[i] + SPACE_OF_MAIN_THREAD;
    }
    start_addr = (char *)pmem[0];
    curr_addr = start_addr;
    

    btree *bt; 
    bt = new btree(); 

    std::vector<uint64_t> keys(num_puts), values(num_puts); 
    for(uint64_t i = 0; i < num_puts; ++i) {
        keys[i] = rand()%10000000;
        values[i] = rand()%10000000;
    } 
 
    std::cout << "keys generated\n"; 
    auto start = TIME_NOW; 
#pragma omp parallel for num_threads(nthreads) 
    for(uint64_t i = 0; i < num_puts; ++i) {
        bt->insert(keys[i], (char*)values[i]); 
        //std::cout << "i: " << i << "\t"; 
    }
    auto insert_time = (TIME_NOW - start).count(); 
    std::cout << "insert_time: " << insert_time/1000000.0 << "\n"; 
    insert_time = insert_time/1000000.0;
    std::cout << "throughput: " << float(num_puts)/insert_time/1000.0 << " Mops/sec\n"; 

    return 0;
}
