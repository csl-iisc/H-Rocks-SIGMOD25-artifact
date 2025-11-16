// Simple atomic transaction benchmark: sequential puts then gets.
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <unistd.h>   // getopt

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
    << " -n <num_ops> -b <batch_size> -k <key_bytes> -v <value_bytes> -f <db_path>\n"
    << " num_ops is the total operations (puts+gets); puts = gets = num_ops/2\n"
    << " batch_size is accepted for interface parity but not used by the RocksDB path\n";
}

int main(int argc, char** argv) {
  uint64_t num_ops   = 0;
  uint64_t batch_sz  = 0;
  size_t   key_size  = 0;
  size_t   val_size  = 0;
  std::string db_path;

  int opt;
  while ((opt = getopt(argc, argv, "n:b:k:v:f:")) != -1) {
    switch (opt) {
      case 'n': num_ops  = std::strtoull(optarg, nullptr, 10); break;
      case 'b': batch_sz = std::strtoull(optarg, nullptr, 10); break;
      case 'k': key_size = std::strtoul (optarg, nullptr, 10); break;
      case 'v': val_size = std::strtoul (optarg, nullptr, 10); break;
      case 'f': db_path  = optarg; break;
      default: usage(argv[0]); return 2;
    }
  }

  if (num_ops == 0 || key_size == 0 || val_size == 0 || db_path.empty()) {
    usage(argv[0]);
    return 2;
  }

  uint64_t num_puts = num_ops / 2;

  std::cout << "Number of puts: " << num_puts   << "\n"
            << "Key size: "       << key_size   << "\n"
            << "Value size: "     << val_size   << "\n"
            << "Batch size: "     << batch_sz   << "\n"
            << "DB path: "        << db_path    << "\n";

  rocksdb::Options options;
  options.create_if_missing = true;
  options.IncreaseParallelism();
  options.OptimizeLevelStyleCompaction();
  options.write_buffer_size = 512ull * 1024 * 1024; // 512 MiB

  rocksdb::DB* db = nullptr;
  auto s = rocksdb::DB::Open(options, db_path, &db);
  if (!s.ok()) {
    std::cerr << "DB::Open failed: " << s.ToString() << "\n";
    return 1;
  }

  std::mt19937_64 rng(std::random_device{}());
  std::vector<std::string> keys(num_puts);
  std::vector<std::string> vals(num_puts);
  for (uint64_t i = 0; i < num_puts; ++i) {
    keys[i] = random_string(key_size, rng);
    vals[i] = random_string(val_size, rng);
  }

  // PUT phase
  auto t0 = Clock::now();
  for (uint64_t i = 0; i < num_puts; ++i) {
    s = db->Put(rocksdb::WriteOptions(), keys[i], vals[i]);
    if (!s.ok()) {
      std::cerr << "Put failed at i=" << i << ": " << s.ToString() << "\n";
      return 1;
    }
  }
  double put_ms = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count() / 1000.0;
  double put_thr = (put_ms > 0.0) ? (num_puts * 1000.0 / put_ms) : 0.0;
  std::cout << "put_time_ms: " << put_ms << " | throughput_ops_per_s: " << put_thr << "\n";

  // GET phase
  t0 = Clock::now();
  std::string out;
  for (uint64_t i = 0; i < num_puts; ++i) {
    s = db->Get(rocksdb::ReadOptions(), keys[i], &out);
    if (!s.ok()) {
      std::cerr << "Get failed at i=" << i << ": " << s.ToString() << "\n";
      return 1;
    }
  }
  double get_ms = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count() / 1000.0;
  double get_thr = (get_ms > 0.0) ? (num_puts * 1000.0 / get_ms) : 0.0;
  std::cout << "get_time_ms: " << get_ms << " | throughput_ops_per_s: " << get_thr << "\n";

  delete db;
  return 0;
}
