// test_range.cc
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
#include <memory>
#include <unistd.h>   // getopt
#ifdef _OPENMP
#include <omp.h>
#endif

#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/iterator.h>

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
    << " -p <prefill> -q <num_ranges> -L <range_len>"
       " -k <key_bytes> -v <value_bytes> -f <db_path>"
       " [-t <threads>] [-r] [-s <seed>]\n";
}

int main(int argc, char** argv) {
  uint64_t prefill     = 0;   // -p
  uint64_t num_ranges  = 0;   // -q
  uint64_t range_len   = 0;   // -L (number of keys per range)
  size_t   key_size    = 0;   // -k
  size_t   value_size  = 0;   // -v
  std::string db_path;        // -f
  int nthreads = 1;           // -t
  bool randomize = false;     // -r
  uint64_t seed = std::random_device{}(); // -s

  int opt;
  while ((opt = getopt(argc, argv, "p:q:L:k:v:f:t:rs:")) != -1) {
    switch (opt) {
      case 'p': prefill     = std::strtoull(optarg, nullptr, 10); break;
      case 'q': num_ranges  = std::strtoull(optarg, nullptr, 10); break;
      case 'L': range_len   = std::strtoull(optarg, nullptr, 10); break;
      case 'k': key_size    = std::strtoul (optarg, nullptr, 10); break;
      case 'v': value_size  = std::strtoul (optarg, nullptr, 10); break;
      case 'f': db_path     = optarg; break;
      case 't': nthreads    = std::max(1, std::atoi(optarg)); break;
      case 'r': randomize   = true; break;
      case 's': seed        = std::strtoull(optarg, nullptr, 10); break;
      default: usage(argv[0]); return 2;
    }
  }
  if (prefill == 0 || num_ranges == 0 || range_len == 0 ||
      key_size == 0 || value_size == 0 || db_path.empty()) {
    usage(argv[0]); return 2;
  }

  std::cout << "DB path: "       << db_path    << "\n"
            << "Prefill: "       << prefill    << "\n"
            << "Ranges: "        << num_ranges << "\n"
            << "Range len: "     << range_len  << "\n"
            << "Key size: "      << key_size   << "\n"
            << "Value size: "    << value_size << "\n"
            << "Threads: "       << nthreads   << "\n"
            << "Randomize: "     << (randomize ? "yes" : "no") << "\n"
            << "Seed: "          << seed       << "\n";

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

  // Prepare keys & values
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
      std::mt19937_64 rng_key((0x9e3779b97f4a7c15ULL ^ i) + (i << 7));
      std::mt19937_64 rng_val((0xbf58476d1ce4e5b9ULL ^ i) + (i << 9));
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

  // Choose starting points for ranges (indices into keys[])
  std::vector<uint64_t> starts;
  starts.reserve(num_ranges);
  if (randomize) {
    std::mt19937_64 rng(seed);
    if (num_ranges <= prefill) {
      starts.resize(prefill);
      std::iota(starts.begin(), starts.end(), 0ull);
      std::shuffle(starts.begin(), starts.end(), rng);
      starts.resize(num_ranges);
    } else {
      std::uniform_int_distribution<uint64_t> dist(0ull, prefill - 1);
      starts.resize(num_ranges);
      for (uint64_t i = 0; i < num_ranges; ++i) starts[i] = dist(rng);
    }
  } else {
    starts.resize(std::min<uint64_t>(num_ranges, prefill));
    std::iota(starts.begin(), starts.end(), 0ull);
    if (num_ranges > prefill) {
      uint64_t need = num_ranges - prefill;
      starts.reserve(num_ranges);
      for (uint64_t i = 0; i < need; ++i) starts.push_back(i % prefill);
    }
  }

  // Range scans: for each start key, Seek() then Next() up to range_len keys
  std::cout << "***************** RANGES *****************\n";
  std::atomic<uint64_t> keys_read{0};
  std::atomic<uint64_t> bytes_read{0};
  std::atomic<uint64_t> ok_ranges{0};

  rocksdb::ReadOptions ro;       // you can tweak (e.g., verify_checksums=false)
  ro.fill_cache = true;          // default; set false for pure scan w/o warming cache

  auto t_rng0 = Clock::now();
#ifdef _OPENMP
#pragma omp parallel num_threads(nthreads)
  {
    std::unique_ptr<rocksdb::Iterator> it(db->NewIterator(ro));
    uint64_t local_keys = 0, local_bytes = 0, local_ranges = 0;

#pragma omp for schedule(static)
    for (uint64_t i = 0; i < num_ranges; ++i) {
      const uint64_t idx = starts[i];
      it->Seek(keys[idx]);
      uint64_t taken = 0;
      while (it->Valid() && taken < range_len) {
        local_keys++;
        local_bytes += it->value().size();
        it->Next();
        ++taken;
      }
      if (!it->status().ok()) {
#pragma omp critical
        std::cerr << "Iterator status error: " << it->status().ToString() << "\n";
      } else {
        local_ranges++;
      }
    }

#pragma omp atomic
    keys_read += local_keys;
#pragma omp atomic
    bytes_read += local_bytes;
#pragma omp atomic
    ok_ranges += local_ranges;
  }
#else
  {
    std::unique_ptr<rocksdb::Iterator> it(db->NewIterator(ro));
    uint64_t local_ranges = 0;
    for (uint64_t i = 0; i < num_ranges; ++i) {
      const uint64_t idx = starts[i];
      it->Seek(keys[idx]);
      uint64_t taken = 0;
      while (it->Valid() && taken < range_len) {
        keys_read.fetch_add(1, std::memory_order_relaxed);
        bytes_read.fetch_add(it->value().size(), std::memory_order_relaxed);
        it->Next();
        ++taken;
      }
      if (!it->status().ok()) {
        std::cerr << "Iterator status error: " << it->status().ToString() << "\n";
      } else {
        ok_ranges.fetch_add(1, std::memory_order_relaxed);
      }
    }
  }
#endif
  auto rng_ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t_rng0).count();

  double rng_sec = rng_ms / 1000.0;
  double key_mops = rng_sec > 0 ? (static_cast<double>(keys_read.load()) / rng_sec) / 1e6 : 0.0;
  double rng_kps  = rng_sec > 0 ? (static_cast<double>(ok_ranges.load()) / rng_sec) : 0.0;

  std::cout << "range_time: " << rng_ms << " ms | key_throughput: "
            << key_mops << " Mkeys/s | range_rate: " << rng_kps << " ranges/s\n"
            << "keys_read: " << keys_read.load()
            << " | bytes_read: " << bytes_read.load()
            << " | avg_keys_per_range: "
            << (ok_ranges.load() ? (keys_read.load() / ok_ranges.load()) : 0) << "\n"
            << "prep_ms: " << prep_ms << " | open_ms: " << open_ms
            << " | prefill_ms: " << put_ms << " | range_ms: " << rng_ms << "\n";

  delete db;
  return 0;
}
