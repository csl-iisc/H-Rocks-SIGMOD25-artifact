// test_ycsbB.cc  —  90% reads, 10% puts (inserts), no updates
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
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

// ------------------------------ Key/value helpers ------------------------------
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

// make a deterministic, unique key from an integer id (base-62, right-aligned)
static std::string make_key_from_id(uint64_t id, size_t key_size) {
  std::string s(key_size, '0');
  size_t pos = key_size;
  while (pos > 0) {
    s[--pos] = CHARSET[id % 62];
    id /= 62;
    if (id == 0) break;
  }
  return s;
}

static inline uint64_t mix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  x = x ^ (x >> 31);
  return x;
}

// --------------------------- Scrambled Zipfian (YCSB-style) ---------------------------
struct ScrambledZipf {
  uint64_t M;       // sampler domain size (cap to keep init fast)
  double   theta;   // skew
  double   zetan;   // zeta(M,theta)
  double   zeta2th; // zeta(2,theta)
  double   alpha;   // 1/(1-theta)
  double   eta;
  uint64_t prefill;

  explicit ScrambledZipf(uint64_t prefill_, double theta_=0.99, uint64_t cap=(1ull<<22))
      : theta(theta_), prefill(prefill_) {
    M = std::min(prefill_, cap ? cap : prefill_);
    zeta2th = 1.0 + std::pow(2.0, -theta);
    zetan = 0.0;
    for (uint64_t i = 1; i <= M; ++i) zetan += 1.0 / std::pow((double)i, theta);
    alpha = 1.0 / (1.0 - theta);
    eta   = (1.0 - std::pow(2.0 / (double)M, 1.0 - theta)) /
            (1.0 - (zeta2th / zetan));
  }

  template <class RNG>
  inline uint64_t next(RNG& rng) const {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    double u  = U(rng);
    double uz = u * zetan;

    uint64_t rank;
    if (uz < 1.0) {
      rank = 0;
    } else if (uz < 1.0 + std::pow(0.5, theta)) {
      rank = 1;
    } else {
      rank = (uint64_t)((double)M * std::pow(eta * u - eta + 1.0, alpha));
      if (rank >= M) rank = M - 1;
    }
    return mix64(rank) % prefill;  // spread over whole [0..prefill-1]
  }
};

// -------------------------------- CLI --------------------------------
static void usage(const char* prog) {
  std::cerr
    << "YCSB-B (90% reads, 10% puts/inserts)\n"
    << "Usage: " << prog
    << " -p <prefill> -n <ops> -k <key_bytes> -v <value_bytes> -f <db_path>\n"
    << "             [-t <threads>] [-z <zipf_theta>] [-U] [-s <seed>]\n"
    << "  -z <theta> : Zipfian skew (default 0.99)\n"
    << "  -U         : Use uniform key selection for reads (instead of Zipf)\n"
    << "  -s <seed>  : RNG seed (default: random)\n";
}

int main(int argc, char** argv) {
  uint64_t prefill    = 0;    // -p
  uint64_t ops        = 0;    // -n
  size_t   key_size   = 0;    // -k
  size_t   value_size = 0;    // -v
  std::string db_path;        // -f
  int nthreads = 1;           // -t
  double zipf_theta = 0.99;   // -z (community default)
  bool use_uniform = false;   // -U (for reads)
  uint64_t seed = std::random_device{}(); // -s

  int opt;
  while ((opt = getopt(argc, argv, "p:n:k:v:f:t:z:Us:")) != -1) {
    switch (opt) {
      case 'p': prefill    = std::strtoull(optarg, nullptr, 10); break;
      case 'n': ops        = std::strtoull(optarg, nullptr, 10); break;
      case 'k': key_size   = std::strtoul (optarg, nullptr, 10); break;
      case 'v': value_size = std::strtoul (optarg, nullptr, 10); break;
      case 'f': db_path    = optarg; break;
      case 't': nthreads   = std::max(1, std::atoi(optarg)); break;
      case 'z': zipf_theta = std::atof(optarg); break;
      case 'U': use_uniform= true; break;
      case 's': seed       = std::strtoull(optarg, nullptr, 10); break;
      default: usage(argv[0]); return 2;
    }
  }
  if (prefill == 0 || ops == 0 || key_size == 0 || value_size == 0 || db_path.empty()) {
    usage(argv[0]); return 2;
  }

  // Split ops: 90% reads, 10% inserts
  uint64_t reads  = (ops * 9) / 10;
  uint64_t writes = ops - reads;

  std::cout << "DB path: "     << db_path    << "\n"
            << "Prefill: "     << prefill    << "\n"
            << "Ops: "         << ops        << " (reads=" << reads << ", puts=" << writes << ")\n"
            << "Key size: "    << key_size   << "\n"
            << "Value size: "  << value_size << "\n"
            << "Threads: "     << nthreads   << "\n"
            << "Read dist: "   << (use_uniform ? "uniform" : "zipfian") << "\n"
            << "Zipf theta: "  << zipf_theta << "\n"
            << "Seed: "        << seed       << "\n";

  // ------------------------- RocksDB options -------------------------
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

  // ---------------------- Prefill keys & values ----------------------
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

  // ---------------------- Build op schedule --------------------------
  std::vector<uint8_t> is_read(ops, 0);
  for (uint64_t i = 0; i < reads; ++i) is_read[i] = 1;
  {
    std::mt19937_64 rng(seed ^ 0xA5A5A5A5A5A5A5A5ull);
    std::shuffle(is_read.begin(), is_read.end(), rng);
  }

  // Samplers
  ScrambledZipf zipf(prefill, (use_uniform ? 0.99 : zipf_theta)); // theta ignored when uniform
  std::uniform_int_distribution<uint64_t> U(0, prefill - 1);

  std::atomic<uint64_t> read_ok{0}, read_miss{0}, put_ok{0};

  // Unique key ids for inserts start after prefill and keep increasing
  std::atomic<uint64_t> next_id{prefill};

  // -------------------- Execute mixed operations ---------------------
  std::cout << "******************* RUN *******************\n";
  auto t_run0 = Clock::now();

#ifdef _OPENMP
#pragma omp parallel num_threads(nthreads)
  {
    std::mt19937_64 rng(seed ^ mix64(
#ifdef _OPENMP
        (uint64_t)omp_get_thread_num()
#else
        0ull
#endif
      ));
#pragma omp for schedule(static)
    for (uint64_t i = 0; i < ops; ++i) {
      if (is_read[i]) {
        // READ: sample from initial keyspace only
        uint64_t idx = use_uniform ? U(rng) : zipf.next(rng);
        std::string val;
        rocksdb::Status s = db->Get(rocksdb::ReadOptions(), keys[idx], &val);
        if (s.ok()) {
          read_ok.fetch_add(1, std::memory_order_relaxed);
        } else if (s.IsNotFound()) {
          read_miss.fetch_add(1, std::memory_order_relaxed);
        } else {
#pragma omp critical
          std::cerr << "Get error: " << s.ToString() << "\n";
        }
      } else {
        // PUT (insert): generate a new unique key from an ever-increasing id
        uint64_t id = next_id.fetch_add(1, std::memory_order_relaxed);
        std::string k = make_key_from_id(id, key_size);
        std::string v = random_string(value_size, rng);
        rocksdb::Status s = db->Put(rocksdb::WriteOptions(), k, v);
        if (s.ok()) {
          put_ok.fetch_add(1, std::memory_order_relaxed);
        } else {
#pragma omp critical
          std::cerr << "Put error: " << s.ToString() << "\n";
        }
      }
    }
  }
#else
  {
    std::mt19937_64 rng(seed);
    for (uint64_t i = 0; i < ops; ++i) {
      if (is_read[i]) {
        uint64_t idx = use_uniform ? U(rng) : zipf.next(rng);
        std::string val;
        rocksdb::Status s = db->Get(rocksdb::ReadOptions(), keys[idx], &val);
        if (s.ok()) read_ok.fetch_add(1, std::memory_order_relaxed);
        else if (s.IsNotFound()) read_miss.fetch_add(1, std::memory_order_relaxed);
        else std::cerr << "Get error: " << s.ToString() << "\n";
      } else {
        uint64_t id = next_id.fetch_add(1, std::memory_order_relaxed);
        std::string k = make_key_from_id(id, key_size);
        std::string v = random_string(value_size, rng);
        rocksdb::Status s = db->Put(rocksdb::WriteOptions(), k, v);
        if (s.ok()) put_ok.fetch_add(1, std::memory_order_relaxed);
        else std::cerr << "Put error: " << s.ToString() << "\n";
      }
    }
  }
#endif

  auto run_ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t_run0).count();
  double run_sec = run_ms / 1000.0;
  double ops_mops = run_sec > 0 ? (ops / run_sec) / 1e6 : 0.0;

  // ----------------------------- Report ------------------------------
  std::cout << "run_time: " << run_ms << " ms | throughput: " << ops_mops << " Mops/s\n"
            << "reads_ok: " << read_ok.load() << " | reads_miss: " << read_miss.load()
            << " | puts_ok: " << put_ok.load() << "\n"
            << "open_ms: " << open_ms << " | prep_ms: " << prep_ms
            << " | prefill_ms: " << put_ms << " | run_ms: " << run_ms << "\n";

  delete db;
  return 0;
}
