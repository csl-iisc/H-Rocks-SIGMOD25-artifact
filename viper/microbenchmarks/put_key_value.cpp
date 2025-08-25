#include "benchmark/benchmark.h"
#include "viper/viper.hpp"
#include <iostream>
#include <cstdio>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <functional>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <getopt.h>
#include <unistd.h>
#include <mutex>
#include <pthread.h>

// Defining macros for timing analysis
#define TIME_NOW std::chrono::high_resolution_clock::now()
#define time_diff(a,b) std::chrono::duration_cast<std::chrono::microseconds>(a-b).count()

using namespace std;

typedef std::vector<char> char_array;

int n = 100000000;
int num_total_prefill = 0;

// int n = 5000;
// int num_total_prefill = 10000;

size_t key_size=8, value_size=16;      // Fix key value size

std::vector<std::string> keys(n);
std::vector<std::string> values(n);

static constexpr uint64_t ONE_GB = (1024ul*1024*1024) * 1;  // 1GB
const size_t initial_size = ONE_GB*16;  // 16 GiB

auto viper_db = viper::Viper<std::string, std::string>::create("/pmem/viper_put_kv", initial_size);

char_array charset()
{
    //Change this to suit
    return char_array( 
            {'0','1','2','3','4',
            '5','6','7','8','9',
            'A','B','C','D','E','F',
            'G','H','I','J','K',
            'L','M','N','O','P',
            'Q','R','S','T','U',
            'V','W','X','Y','Z',
            'a','b','c','d','e','f',
            'g','h','i','j','k',
            'l','m','n','o','p',
            'q','r','s','t','u',
            'v','w','x','y','z'
            });
};   

class MyFixture : public benchmark::Fixture
{
    public:
        void SetUp(benchmark::State& state) override {}
        void TearDown(benchmark::State& state) override {}

        int num_total_prefill = 0;
        int n = 100000000;

        std::string generate_random_string(size_t length, std::function<char(void)> rand_char){
            std::string str(length,0);
            std::generate_n( str.begin(), length, rand_char );
            return str;
        }

        inline void prefill() {
            // To modify records in Viper, you need to use a Viper Client.
            auto v_client = viper_db->get_client();

            for (uint64_t i = 0; i < num_total_prefill; ++i) {
                std::string value = values[i];
                std::string key = keys[i];
                v_client.put(key, value);
            }

        }

        void initialize_array(const uint64_t n1, const uint64_t n2) {
            // Setting up parameters for random string generation
            const auto ch_set = charset();
            // Using a non-deterministic random number generator
            std::default_random_engine rng(std::random_device{}());
            // Creates uniformly distributed indices into the character set
            std::uniform_int_distribution<> dist(0, ch_set.size()-1);
            // Function that ties the distribution and generator together
            auto randchar = [ch_set, &dist, &rng](){return ch_set[ dist(rng) ];};
            int i=n1;

            while(i<n2) {
                keys[i] = generate_random_string(key_size,randchar);
                values[i] = generate_random_string(value_size,randchar);
                ++i;
            }
        }

        MyFixture()
        {
            std::cout << "Called once per fixture testcase" << std::endl;
            // call whatever setup functions you need in the fixtures ctor 
            initialize_array(0, num_total_prefill);
            prefill();
            std::cout << "Done prefilling" <<std::endl;
            initialize_array(0, n);
        }
};

BENCHMARK_DEFINE_F(MyFixture, bm_put)(benchmark::State& state){
    const uint64_t num_total_prefill = state.range(0);
    const uint64_t num_total_inserts = state.range(1);

    // set_cpu_affinity(state.thread_index);

    const uint64_t num_inserts_per_thread = num_total_inserts / state.threads();
    const uint64_t start_idx = (state.thread_index() * num_inserts_per_thread);
    const uint64_t end_idx = start_idx + num_inserts_per_thread;

    for (auto _ : state) {  
        // To modify records in Viper, you need to use a Viper Client.
        auto v_client = viper_db->get_client();
        for (uint64_t i = start_idx; i < end_idx; ++i) {
            std::string value = values[i];
            std::string key = keys[i];
            v_client.put(key, value);
        }
    }
    state.SetItemsProcessed(num_inserts_per_thread);
}

int main(int argc, char** argv) 
{
    int option_char;
    size_t key_size, value_size;
    uint64_t num_keys; 
    std::string file_loc; 

    while ((option_char = getopt (argc, argv, ":n:k:v:")) != -1){
        switch (option_char)
        {
            case 'n': num_keys = atoi (optarg); break;
            case 'k': key_size = atoi (optarg); break;
            case 'v': value_size = atoi (optarg); break;
            case ':': fprintf (stderr, "option needs a value\n");
            case '?': fprintf (stderr, "usage: %s [-n <number of keys>] [-k <key size>] [-v <value size>]\n", argv[0]);
        }
    }

    std::cout << "Number of keys: " << n << std::endl;
    std::cout << "Key size: " << key_size << std::endl;
    std::cout << "Value size: " << value_size << std::endl;


    BENCHMARK_REGISTER_F(MyFixture, bm_put)->Threads(128);
    //BENCHMARK_MAIN();
}
