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

int num_total_prefill = 50000000;
// int num_total_prefill = 15000;
size_t key_size=8, value_size=8;      // Fix key value size
int r = 10000000;

std::vector<std::string> keys(num_total_prefill);
std::vector<std::string> values(num_total_prefill);

static constexpr uint64_t ONE_GB = (1024ul*1024*1024) * 1;  // 1GB
const size_t initial_size = ONE_GB;  // 16 GiB

auto viper_db = viper::Viper<std::string, std::string>::create("/pmem/viper_get_kv", initial_size);

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

    int num_total_prefill = 50000000;

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
        const auto ch_set = charset();
        std::default_random_engine rng(std::random_device{}());
        std::uniform_int_distribution<> dist(0, ch_set.size()-1);
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
    }
};

BENCHMARK_DEFINE_F(MyFixture, bm_get)(benchmark::State& state){
    //int r = state.range(0);
    uint64_t num_gets = state.range(0); 
    cout << num_gets; 
    //uint64_t num_finds = (num_total_prefill)*r/100;
    uint64_t found_counter = 0;
    const uint64_t num_gets_per_thread = num_gets / state.threads();

    auto start_time = TIME_NOW;
    for (auto _ : state) {
        const uint64_t start_idx = (state.thread_index() * num_gets_per_thread);
        const uint64_t end_idx = start_idx + num_gets_per_thread;
        std::random_device rnd{};
        auto rnd_engine = std::default_random_engine(rnd());
        std::uniform_int_distribution<> distrib(start_idx, end_idx);

        const auto v_client = viper_db->get_read_only_client();
        
        string value;
        for (uint64_t i = 0; i < num_gets_per_thread; ++i) {
            const uint64_t key = distrib(rnd_engine);
            const string db_key = keys[key];
            const bool found = v_client.get(db_key, &value);
            assert(found && (value == values[key]));
        }
    }
    // std::cout<<"Found matches = "<<found_counter<<std::endl;
    auto time_taken = time_diff(TIME_NOW, start_time)/1000000.0f;
    // std::cout<<"Time Taken for adding reading "<<r<<"% keys = "<<time_taken<<std::endl;
}

//BENCHMARK_REGISTER_F(MyFixture, bm_get)->Threads(128);
BENCHMARK_REGISTER_F(MyFixture, bm_get)->Args({128})->Args({r});
// BENCHMARK_F(MyFixture, bm_get)->ThreadRange(1, 128)->Args({r});

BENCHMARK_MAIN();
