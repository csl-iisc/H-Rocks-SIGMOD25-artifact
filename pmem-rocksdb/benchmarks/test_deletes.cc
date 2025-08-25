// test_deletes.cc
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
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
  std::string s;
  s.resize(len);
  for (size_t i = 0; i < len; ++i) s[i] = CHARSET[dist(rng)];
  return s;
}

static void usage(const char* prog) {
  std::cerr
    << "Usage: " << prog
    << " -p <prefill> -n <num_deletes> -k <key_bytes> -v <value_bytes>"
       " -f <db_path> [-t <threads>]\n";
}

int main(int argc, char** argv) {
  uint64_t prefill    = 0;   // -p
  uint64_t num_del    = 0;   // -n
  size_t   key_size   = 0;   // -k
  size_t   value_size = 0;   // -v
  std::string db_path;       // -f
  int nthreads = 1;          // -t

  int opt;
  while ((opt = getopt(argc, argv, "p:n:k:v:f:t:")) != -1) {
    switch (opt) {
      case 'p': prefill    = std::strtoull(optarg, nullptr, 10); break;
      case 'n': num_del    = std::strtoull(optarg, nullptr, 10); break;
      case 'k': key_size   = std::strtoul (optarg, nullptr, 10); break;
      case 'v': value_size = std::strtoul (optarg, nullptr, 10); break;
      case 'f': db_path    = optarg; break;
      case 't': nthreads   = std::max(1, std::atoi(optarg)); break;
      default: usage(argv[0]); return 2;
    }
  }
  if (prefill == 0 || key_size == 0 || value_size == 0 || db_path.empty()) {
    usage(argv[0]); return 2;
  }
  if (num_del == 0) num_del = prefill; // default: delete everything we inserted

  std::cout << "DB path: "      << db_path    << "\n"
            << "Prefill: "      << prefill    << "\n"
            << "Deletes: "      << num_del    << "\n"
            << "Key size: "     << key_size   << "\n"
            << "Value size: "   << value_size << "\n"
            << "Threads: "      << nthreads   << "\n";

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

  // Prepare keys/values
  std::vector<std::string> keys(prefill);
  std::vector<std::string> vals(prefill);

  auto t_prep0 = Clock::now();
#ifdef _OPENMP
  omp_set_num_threads(nthreads);
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
  std::mt19937_64 rng_key(1234567ULL), rng_val(7654321ULL);
  for (uint64_t i = 0; i < prefill; ++i) {
    keys[i] = random_string(key_size, rng_key);
    vals[i] = random_string(value_size, rng_val);
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

  // Deletes (use first min(num_del, prefill) keys)
  uint64_t del_count = std::min<uint64_t>(num_del, prefill);
  std::cout << "***************** DELETES ****************\n";
  auto t_del0 = Clock::now();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nthreads)
  for (uint64_t i = 0; i < del_count; ++i) {
    rocksdb::Status s = db->Delete(rocksdb::WriteOptions(), keys[i]);
    if (!s.ok()) {
#pragma omp critical
      std::cerr << "Delete failed at i=" << i << ": " << s.ToString() << "\n";
    }
  }
#else
  for (uint64_t i = 0; i < del_count; ++i) {
    rocksdb::Status s = db->Delete(rocksdb::WriteOptions(), keys[i]);
    if (!s.ok()) std::cerr << "Delete failed at i=" << i << ": " << s.ToString() << "\n";
  }
#endif
  db->Flush(rocksdb::FlushOptions());
  auto del_ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t_del0).count();
  double del_sec = del_ms / 1000.0;
  double del_mops = del_sec > 0 ? (del_count / del_sec) / 1e6 : 0.0;
  std::cout << "delete_time: " << del_ms << " ms | throughput: " << del_mops << " Mops/s\n";

  // Summary
  std::cout << "prep_ms: " << prep_ms << " | open_ms: " << open_ms
            << " | prefill_ms: " << put_ms << " | delete_ms: " << del_ms << "\n";

  delete db;
  return 0;
}
