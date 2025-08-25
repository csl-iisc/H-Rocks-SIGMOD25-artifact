#include "batch.h"
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
    std::string file_name;
    while ((option_char = getopt (argc, argv, ":f:")) != -1) {
        switch (option_char)
        {
            case 'f': file_name = atoi (optarg); break;
            case ':': fprintf (stderr, "option needs a value\n");
            case '?': fprintf (stderr, "usage: %s [-n <number of keys>] [-k <key size>] [-v     <value size>]\n", argv[0]);
        }
    }

    rocksdb::DB *db; 
    rocksdb::Options options; 
    options.IncreaseParallelism(64); 
    options.create_if_missing = true; 
    rocksdb::Status s = rocksdb::DB::Open(options, "/pmem/rdb_ycsb", &db); 
    assert(s.ok()); 
    std::cout << "DB opened\n"; 

    std::vector<Command> readCommands, writeCommands, updateCommands;
    Batch batch(readCommands, writeCommands, updateCommands, 0, db);
    // Perform a put operation

    std::string data_loc = "YCSB_data/" + file_name; 
    cout << data_loc << "\n"; 
    std::ifstream file(data_loc);
    std::string line;
    std::regex insert_regex(R"(INSERT usertable (user\d+) \[ field0=(.+?) \])");
    std::regex read_regex(R"(READ usertable (user\d+) \[ field0 \])");


    if (file.is_open()) {
        while (std::getline(file, line)) {
            std::smatch match;
            if (std::regex_search(line, match, insert_regex) && match.size() > 2) {
                std::string user_key = match.str(1);
                std::string field_value = match.str(2);
                batch.Put(user_key, field_value); 
            } else if (std::regex_search(line, match, read_regex) && match.size() > 1) {
                std::string user_key = match.str(1);
                std::string value; 
                //std::cout << "READ: " << user_key << std::endl;
                batch.Get(user_key); 
            }
        }
        file.close();
    } else {
        std::cout << "Unable to open file";
    }
 
    // Perform a get operation
    batch.Exit(); 
    return 0; 
}
