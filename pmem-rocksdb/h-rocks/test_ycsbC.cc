#include "batch.h"
#include "pmem_paths.h"
#include <iostream> 
#include <cstdio>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <functional>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <getopt.h>
#include <unistd.h>
#include <algorithm>  //for std::generate_n
#include <set>

#include <iostream>
#include <fstream>

#include <bits/stdc++.h>
#include <rocksdb/db.h>
#include <rocksdb/slice.h>
#include <rocksdb/options.h>
#include <rocksdb/write_batch.h>
#include "rocksdb/statistics.h"

using namespace std; 
using namespace rocksdb; 

#define TIME_NOW std::chrono::high_resolution_clock::now()
#define  FALSE          0       // Boolean false
#define  TRUE           1       // Boolean true

double rand_val(int seed)
{
    const long  a =      16807;  // Multiplier
    const long  m = 2147483647;  // Modulus
    const long  q =     127773;  // m div a
    const long  r =       2836;  // m mod a
    static long x;               // Random int value
    long        x_div_q;         // x divided by q
    long        x_mod_q;         // x modulo q
    long        x_new;           // New x value

    // Set the seed if argument is non-zero and then return zero
    if (seed > 0)
    {
        x = seed;
        return(0.0);
    }

    // RNG using integer arithmetic
    x_div_q = x / q;
    x_mod_q = x % q;
    x_new = (a * x_mod_q) - (r * x_div_q);
    if (x_new > 0)
        x = x_new;
    else
        x = x_new + m;

    // Return a random value between 0.0 and 1.0
    return((double) x / m);
}

int zipf(double alpha, int n)
{
    static int first = TRUE;      // Static first time flag
    static double c = 0;          // Normalization constant
    static double *sum_probs;     // Pre-calculated sum of probabilities
    double z;                     // Uniform random number (0 < z < 1)
    int zipf_value;               // Computed exponential value to be returned
    int    i;                     // Loop counter
    int low, high, mid;           // Binary-search bounds

    // Compute normalization constant on first call only
    if (first == TRUE)
    {
        for (i=1; i<=n; i++)
            c = c + (1.0 / pow((double) i, alpha));
        c = 1.0 / c;

        sum_probs = (double*)malloc((n+1)*sizeof(*sum_probs));
        sum_probs[0] = 0;
        for (i=1; i<=n; i++) {
            sum_probs[i] = sum_probs[i-1] + c / pow((double) i, alpha);
        }
        first = FALSE;
    }

    // Pull a uniform random number (0 < z < 1)
    do
    {
        z = rand_val(0);
    }
    while ((z == 0) || (z == 1));

    // Map z to the value
    low = 1, high = n;
    do {
        mid = floor((low+high)/2);
        if (sum_probs[mid] >= z && sum_probs[mid-1] < z) {
            zipf_value = mid;
            break;
        } else if (sum_probs[mid] >= z) {
            high = mid-1;
        } else {
            low = mid+1;
        }
    } while (low <= high);

    // Assert that zipf_value is between 1 and N
    assert((zipf_value >=1) && (zipf_value <= n));

    return(zipf_value);
}



typedef std::vector<char> char_array;

char_array charset()
{
    //Change this to suit
    return 
        char_array({'A','B','C','D','E','F',
                'G','H','I','J','K',
                'L','M','N','O','P',
                'Q','R','S','T','U',
                'V','W','X','Y','Z',
                });
};

std::string generate_random_string(size_t length, std::function<char(void)> rand_char)
{
    std::string str(length,0);
    std::generate_n(str.begin(), length, rand_char);
    return str;
}


int main(int argc, char **argv) 
{
    int option_char;
    uint64_t num_ops, num_puts; 
    size_t key_size, value_size;
    while ((option_char = getopt (argc, argv, ":p:n:k:v:")) != -1) {
        switch (option_char)
        {
            case 'n': num_ops = atoi (optarg); break;
            case 'p': num_puts = atoi (optarg); break;
            case 'k': key_size = atoi (optarg); break;
            case 'v': value_size = atoi (optarg); break;
            case ':': fprintf (stderr, "option needs a value\n");
            case '?': fprintf (stderr, "usage: %s [-n <number of keys>] [-k <key size>] [-v     <value size>]\n", argv[0]);
        }
    }
    //uint64_t num_ops = num_puts + num_gets; 
    uint64_t num_gets = num_ops * 1; 

    std::cout<<"Number of puts: " << num_puts << std::endl;
    std::cout<<"Number of gets: " << num_gets << std::endl;
    std::cout<<"Key size: " << key_size << std::endl;
    std::cout<<"Value size: " << value_size << std::endl;
    rocksdb::DB *db; 
    rocksdb::Options options; 
    options.IncreaseParallelism(64); 
    options.create_if_missing = true; 
    rocksdb::Status s = rocksdb::DB::Open(options, hrocks::PmemPath("rdb_ycsbC"), &db); 
    assert(s.ok()); 
    std::cout << "DB opened\n"; 

    std::vector<uint64_t> key_idx(num_gets) ; 
    rand_val(1); 
    double alpha = 1.0; 
    int zipf_rv;               // Zipf random variable
    cout << "Reached here..\n"; 
    for (uint64_t i = 0; i < num_gets; i++) {
        zipf_rv = zipf(alpha, num_puts);
        key_idx[i] = zipf_rv;  
        //cout << key_idx[i] << "\t"; 
    }

    std::vector<std::string> keys(num_puts); 
    std::vector<std::string> values(num_puts); 
    const auto ch_set = charset();
    std::default_random_engine rng(std::random_device{}());
    std::uniform_int_distribution<> dist(0, ch_set.size()-1);
    auto randchar = [ch_set, &dist, &rng](){return ch_set[dist(rng)];};

    uint64_t puts = 0; 
    for(uint64_t i = 0; i < num_puts; ++i) {
        keys[i] = generate_random_string(key_size - 1, randchar); 
        values[i] = generate_random_string(value_size - 1, randchar); 
    }

    std::vector<Command> readCommands, writeCommands, updateCommands;
    Batch batch(readCommands, writeCommands, updateCommands, 0, db);
    // Perform a put operation
    char *key, *value; 
    key = (char*)malloc(key_size); 
    value = (char*)malloc(value_size); 
    for(uint64_t i = 0; i < num_puts; ++i) {
        strcpy(key, keys[puts].c_str()); 
        strcpy(value, values[puts].c_str()); 
        batch.Put(key, value); 
        puts++; 
    }

    // Perform a get operation
    for(uint64_t i = 0; i < num_gets; ++i) {
        uint64_t get_index = key_idx[i] - 1; 
        //cout << get_index << "\t" << keys[get_index] << "\n"; 
        strcpy(key, keys[get_index].c_str()); 
        batch.Get(key); 
    }

    batch.Exit(); 
    return 0; 
}
