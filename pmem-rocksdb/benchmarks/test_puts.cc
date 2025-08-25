// random_puts_example.cc
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
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
  std::string s;
  s.resize(len);
  for (size_t i = 0; i < len; ++i) s[i] = CHARSET[dist(rng)];
  return s;
}

static void usage(const char* prog) {
  std::cerr
    << "Usage: " << prog
    << " -n <num_puts> -k <key_bytes> -v <value_bytes> -f <db_path> [-t <threads>]\n";
}

int main(int argc, char** argv) {
  uint64_t num_puts   = 0;
  size_t   key_size   = 0;
  size_t   value_size = 0;
  std::string db_path;
  int nthreads = 1;

  int opt;
  while ((opt = getopt(argc, argv, "n:k:v:f:t:")) != -1) {
    switch (opt) {
      case 'n': num_puts   = std::strtoull(optarg, nullptr, 10); break;
      case 'k': key_size   = std::strtoul (optarg, nullptr, 10); break;
      case 'v': value_size = std::strtoul (optarg, nullptr, 10); break;
      case 'f': db_path    = optarg; break;
      case 't': nthreads   = std::max(1, std::atoi(optarg)); break;
      default: usage(argv[0]); return 2;
    }
  }

  if (num_puts == 0 || key_size == 0 || value_size == 0 || db_path.empty()) {
    usage(argv[0]);
    return 2;
  }

  std::cout << "Number of puts: " << num_puts   << "\n"
            << "Key size: "       << key_size   << "\n"
            << "Value size: "     << value_size << "\n"
            << "Threads: "        << nthreads   << "\n"
            << "DB path: "        << db_path    << "\n";

  // RocksDB options
  rocksdb::Options options;
  options.create_if_missing = true;
  options.IncreaseParallelism();               // more compaction threads
  options.OptimizeLevelStyleCompaction();      // reasonable defaults
  options.write_buffer_size = 512ull * 1024 * 1024; // 512 MiB

  // Open DB
  auto t0 = Clock::now();
  rocksdb::DB* db = nullptr;
  auto status = rocksdb::DB::Open(options, db_path, &db);
  if (!status.ok()) {
    std::cerr << "DB::Open failed: " << status.ToString() << "\n";
    return 1;
  }
  auto db_open_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();
  std::cout << "DB opened in " << db_open_ms << " ms\n";

  // Prefill puts
  std::cout << "********************* PREFILL ***************************\n";
  auto t1 = Clock::now();

#ifdef _OPENMP
  omp_set_num_threads(nthreads);
#pragma omp parallel
  {
    std::mt19937_64 rng(
        std::random_device{}() ^
        (static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&rng)) << 1) ^
#ifdef _OPENMP
        static_cast<uint64_t>(omp_get_thread_num())
#else
        0ull
#endif
    );

#pragma omp for schedule(static)
    for (uint64_t i = 0; i < num_puts; ++i) {
      std::string k = random_string(key_size, rng);
      std::string v = random_string(value_size, rng);
      rocksdb::Status st = db->Put(rocksdb::WriteOptions(), k, v);
      if (!st.ok()) {
#pragma omp critical
        std::cerr << "Put failed at i=" << i << ": " << st.ToString() << "\n";
      }
    }
  }
#else
  // Fallback single-threaded if OpenMP is not enabled
  std::mt19937_64 rng(std::random_device{}());
  for (uint64_t i = 0; i < num_puts; ++i) {
    std::string k = random_string(key_size, rng);
    std::string v = random_string(value_size, rng);
    rocksdb::Status st = db->Put(rocksdb::WriteOptions(), k, v);
    if (!st.ok()) {
      std::cerr << "Put failed at i=" << i << ": " << st.ToString() << "\n";
    }
  }
#endif

  // Flush memtables to persist data
  db->Flush(rocksdb::FlushOptions());

  auto elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t1).count();

  // Report
  double seconds = elapsed_ms / 1000.0;
  double mops    = seconds > 0.0 ? (num_puts / seconds) / 1e6 : 0.0;
  std::cout << "prefill_time: " << elapsed_ms << " ms\n"
            << "throughput: "   << mops << " Mops/s\n";

  delete db;
  return 0;
}

