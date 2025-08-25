// test_gets.cc (randomized get order)
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <unistd.h>   // getopt
#ifdef _OPENMP
#include <omp.h>
#endif

#include <rocksdb/db.h>
#include <rocksdb/options.h>

using Clock = std::chrono::high_resolution_clock;

static const char CHARSET[] =
  "0123456789"
  "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
  "abcdefghijklmnopqrstuvwxyz";
static constexpr size_t CHARSET_LEN = sizeof(CHARSET) - 1;

static std::string random_string(size_t len, std::mt19937_64& rng) {
  std::uniform_int_distribution<size_t> dist(0, CHARSET_LEN - 1);
  std::string s(len, '\0');
  for (size_t i = 0; i < len; ++i) s[i] = CHARSET[dist(rng)];
  return s;
}

static void usage(const char* prog) {
  std::cerr
    << "Usage: " << prog
    << " -p <prefill> -g <num_gets> -k <key_bytes> -v <value_bytes>"
       " -f <db_path> [-t <threads>] [-r] [-s <seed>]\n";
}

int main(int argc, char** argv) {
  uint64_t prefill    = 0;   // -p
  uint64_t num_gets   = 0;   // -g
  size_t   key_size   = 0;   // -k
  size_t   value_size = 0;   // -v
  std::string db_path;       // -f
  int nthreads = 1;          // -t
  bool randomize = true;    // -r
  uint64_t seed = std::random_device{}(); // -s

  int opt;
  while ((opt = getopt(argc, argv, "p:g:k:v:f:t:rs:")) != -1) {
    switch (opt) {
      case 'p': prefill    = std::strtoull(optarg, nullptr, 10); break;
      case 'g': num_gets   = std::strtoull(optarg, nullptr, 10); break;
      case 'k': key_size   = std::strtoul (optarg, nullptr, 10); break;
      case 'v': value_size = std::strtoul (optarg, nullptr, 10); break;
      case 'f': db_path    = optarg; break;
      case 't': nthreads   = std::max(1, std::atoi(optarg)); break;
      case 'r': randomize  = true; break;
      case 's': seed       = std::strtoull(optarg, nullptr, 10); break;
      default: usage(argv[0]); return 2;
    }
  }
  if (prefill == 0 || num_gets == 0 || key_size == 0 || value_size == 0 || db_path.empty()) {
    usage(argv[0]); return 2;
  }

  std::cout << "DB path: "      << db_path    << "\n"
            << "Prefill: "      << prefill    << "\n"
            << "Gets: "         << num_gets   << "\n"
            << "Key size: "     << key_size   << "\n"
            << "Value size: "   << value_size << "\n"
            << "Threads: "      << nthreads   << "\n"
            << "Randomize: "    << (randomize ? "yes" : "no") << "\n"
            << "Seed: "         << seed       << "\n";

  // RocksDB options
  rocksdb::Options options;
  options.create_if_missing = true;
  options.IncreaseParallelism();
  options.OptimizeLevelStyleCompaction();
  options.write_buffer_size = 512ull * 1024 * 1024; // 512 MiB

  // Open DB
  auto t_open0 = Clock::now();
  rocksdb::DB* db = nullptr;
  auto st = rocksdb::DB::Open(options, db_path, &db);
  if (!st.ok()) {
    std::cerr << "DB::Open failed: " << st.ToString() << "\n";
    return 1;
  }
  auto open_ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t_open0).count();
  std::cout << "DB opened in " << open_ms << " ms\n";

  // Prepare keys/values for prefill
  std::vector<std::string> keys(prefill);
  std::vector<std::string> vals(prefill);

#ifdef _OPENMP
  omp_set_num_threads(nthreads);
#endif

  auto t_prep0 = Clock::now();
#ifdef _OPENMP
#pragma omp parallel
  {
#pragma omp for schedule(static)
    for (uint64_t i = 0; i < prefill; ++i) {
      std::mt19937_64 rng_key(
          (0x9e3779b97f4a7c15ULL ^ i) + (static_cast<uint64_t>(i) << 7));
      std::mt19937_64 rng_val(
          (0xbf58476d1ce4e5b9ULL ^ i) + (static_cast<uint64_t>(i) << 9));
      keys[i] = random_string(key_size, rng_key);
      vals[i] = random_string(value_size, rng_val);
    }
  }
#else
  {
    std::mt19937_64 rng_key(1234567ULL), rng_val(7654321ULL);
    for (uint64_t i = 0; i < prefill; ++i) {
      keys[i] = random_string(key_size, rng_key);
      vals[i] = random_string(value_size, rng_val);
    }
  }
#endif
  auto prep_ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t_prep0).count();

  // Prefill puts
  std::cout << "************** PREFILL PUTS **************\n";
  auto t_put0 = Clock::now();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nthreads)
  for (uint64_t i = 0; i < prefill; ++i) {
    rocksdb::Status s = db->Put(rocksdb::WriteOptions(), keys[i], vals[i]);
    if (!s.ok()) {
#pragma omp critical
      std::cerr << "Put failed at i=" << i << ": " << s.ToString() << "\n";
    }
  }
#else
  for (uint64_t i = 0; i < prefill; ++i) {
    rocksdb::Status s = db->Put(rocksdb::WriteOptions(), keys[i], vals[i]);
    if (!s.ok()) std::cerr << "Put failed at i=" << i << ": " << s.ToString() << "\n";
  }
#endif
  db->Flush(rocksdb::FlushOptions());
  auto put_ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t_put0).count();
  double put_sec = put_ms / 1000.0;
  double put_mops = put_sec > 0 ? (prefill / put_sec) / 1e6 : 0.0;
  std::cout << "prefill_time: " << put_ms << " ms  | throughput: " << put_mops << " Mops/s\n";

  // Build access order
  uint64_t get_count = std::min<uint64_t>(num_gets, prefill);
  std::vector<uint64_t> order;
  order.reserve(get_count);

  if (randomize) {
    std::mt19937_64 rng(seed);
    if (get_count <= prefill) {
      // Shuffle a permutation and take the first get_count
      order.resize(prefill);
      std::iota(order.begin(), order.end(), 0ull);
      std::shuffle(order.begin(), order.end(), rng);
      order.resize(get_count);
    } else {
      // Sample with replacement
      std::uniform_int_distribution<uint64_t> dist(0ull, prefill - 1);
      order.resize(get_count);
      for (uint64_t i = 0; i < get_count; ++i) order[i] = dist(rng);
    }
  } else {
    // Sequential 0..get_count-1
    order.resize(get_count);
    std::iota(order.begin(), order.end(), 0ull);
  }

  // Gets
  std::cout << "******************* GETS *****************\n";
  std::atomic<uint64_t> ok_count{0}, miss_count{0};

  auto t_get0 = Clock::now();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nthreads)
  for (uint64_t i = 0; i < get_count; ++i) {
    const uint64_t idx = order[i];
    std::string val;
    rocksdb::Status s = db->Get(rocksdb::ReadOptions(), keys[idx], &val);
    if (s.ok()) {
      ok_count.fetch_add(1, std::memory_order_relaxed);
      if (val.size() != value_size) {
#pragma omp critical
        std::cerr << "Warning: value size mismatch at i=" << i
                  << " got=" << val.size() << " exp=" << value_size << "\n";
      }
    } else if (s.IsNotFound()) {
      miss_count.fetch_add(1, std::memory_order_relaxed);
    } else {
#pragma omp critical
      std::cerr << "Get failed at i=" << i << ": " << s.ToString() << "\n";
    }
  }
#else
  for (uint64_t i = 0; i < get_count; ++i) {
    const uint64_t idx = order[i];
    std::string val;
    rocksdb::Status s = db->Get(rocksdb::ReadOptions(), keys[idx], &val);
    if (s.ok()) {
      ok_count.fetch_add(1, std::memory_order_relaxed);
      if (val.size() != value_size) {
        std::cerr << "Warning: value size mismatch at i=" << i
                  << " got=" << val.size() << " exp=" << value_size << "\n";
      }
    } else if (s.IsNotFound()) {
      miss_count.fetch_add(1, std::memory_order_relaxed);
    } else {
      std::cerr << "Get failed at i=" << i << ": " << s.ToString() << "\n";
    }
  }
#endif

  auto get_ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t_get0).count();
  double get_sec = get_ms / 1000.0;
  double get_mops = get_sec > 0 ? (get_count / get_sec) / 1e6 : 0.0;

  std::cout << "get_time: " << get_ms << " ms | throughput: " << get_mops << " Mops/s\n"
            << "hits: " << ok_count.load() << " | misses: " << miss_count.load() << "\n"
            << "prep_ms: " << prep_ms << " | open_ms: " << open_ms
            << " | prefill_ms: " << put_ms << " | get_ms: " << get_ms << "\n";

  delete db;
  return 0;
}
