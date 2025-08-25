// test_ycsbC_plush.cpp — YCSB-C: 100% reads (Zipf α=0.99) over a prefilled set
#include <cassert>
#include <cmath>
#include <chrono>
#include <functional>
#include <getopt.h>
#include <iostream>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>
#include <memory>
#include <span>     // if your toolchain is < C++20, remove and rely on your Hashtable wrapper

#include "../src/hashtable/Hashtable.h"

using namespace std;

#define TIME_NOW std::chrono::high_resolution_clock::now()

// ---------------- RNG + Zipf (same style as your code) ----------------
double rand_val(int seed) {
  const long  a = 16807, m = 2147483647, q = 127773, r = 2836;
  static long x;
  if (seed > 0) { x = seed; return 0.0; }
  long x_div_q = x / q, x_mod_q = x % q;
  long x_new = (a * x_mod_q) - (r * x_div_q);
  x = (x_new > 0) ? x_new : x_new + m;
  return ((double)x / m);
}

int zipf(double alpha, int n) {
  static int first = 1;
  static double c = 0;
  static double *sum_probs;
  if (n <= 0) return 1;

  if (first == 1) {
    for (int i = 1; i <= n; i++) c += 1.0 / std::pow((double)i, alpha);
    c = 1.0 / c;
    sum_probs = (double*)malloc((n + 1) * sizeof(*sum_probs));
    sum_probs[0] = 0;
    for (int i = 1; i <= n; i++) sum_probs[i] = sum_probs[i - 1] + c / std::pow((double)i, alpha);
    first = 0;
  }

  double z;
  do { z = rand_val(0); } while (z == 0 || z == 1);

  int low = 1, high = n, mid, zipf_value = 1;
  while (low <= high) {
    mid = (int)std::floor((low + high) / 2.0);
    if (sum_probs[mid] >= z && sum_probs[mid - 1] < z) { zipf_value = mid; break; }
    else if (sum_probs[mid] >= z) high = mid - 1;
    else low = mid + 1;
  }
  if (zipf_value < 1) zipf_value = 1;
  if (zipf_value > n) zipf_value = n;
  return zipf_value;
}

// ---------------- key/value helpers ----------------
typedef std::vector<char> char_array;
static char_array charset() {
  return char_array({
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'
  });
}
static std::string random_string(size_t length, std::function<char(void)> rand_char) {
  std::string s(length, 0);
  std::generate_n(s.begin(), length, rand_char);
  return s;
}

int main(int argc, char** argv) {
  uint64_t num_ops = 0;         // we will prefill num_ops keys, then do num_ops reads
  size_t key_size = 0, value_size = 0;
  int nthreads = 1;

  int opt;
  while ((opt = getopt(argc, argv, ":n:k:v:t:")) != -1) {
    switch (opt) {
      case 'n': num_ops    = std::strtoull(optarg, nullptr, 10); break;
      case 'k': key_size   = std::strtoul (optarg, nullptr, 10); break;
      case 'v': value_size = std::strtoul (optarg, nullptr, 10); break;
      case 't': nthreads   = std::max(1, atoi(optarg)); break;
      default:
        std::cerr << "usage: " << argv[0]
                  << " -n <ops> -k <key_bytes> -v <value_bytes> -t <threads>\n";
        return 2;
    }
  }

  if (num_ops == 0 || key_size == 0 || value_size == 0) {
    std::cerr << "error: all of -n, -k, -v must be > 0\n";
    return 2;
  }

  std::cout << "YCSB-C | prefill=" << num_ops << " | reads=" << num_ops
            << " | k=" << key_size << " | v=" << value_size
            << " | t=" << nthreads << "\n";

  // Plush hashtable
  const std::string db_path = "/pmem/plush_table";
  Hashtable<std::span<const std::byte>, std::span<const std::byte>, PartitionType::Hash>
      table(db_path, /*create*/ true);

  // RNG for key/value generation
  const auto ch = charset();
  std::default_random_engine rng(std::random_device{}());
  std::uniform_int_distribution<> dist(0, (int)ch.size() - 1);
  auto randchar = [&]() { return ch[dist(rng)]; };

  // Prepare prefill keys/vals
  std::vector<std::string> keys(num_ops);
  std::vector<std::string> vals(num_ops);
  for (uint64_t i = 0; i < num_ops; ++i) {
    keys[i] = random_string(key_size, randchar);
    vals[i] = random_string(value_size, randchar);
  }

  // PREFILL: insert num_ops keys
  auto t_put0 = TIME_NOW;
#pragma omp parallel for num_threads(nthreads) schedule(static)
  for (uint64_t i = 0; i < num_ops; ++i) {
    std::span<const std::byte> k(reinterpret_cast<const std::byte*>(keys[i].data()), keys[i].size());
    std::span<const std::byte> v(reinterpret_cast<const std::byte*>(vals[i].data()),  vals[i].size());
    table.insert(k, v);
  }
  auto put_time_us = std::chrono::duration_cast<std::chrono::microseconds>(TIME_NOW - t_put0).count();
  std::cout << "prefill_time: " << (put_time_us / 1e6) << " s\n";

  // READS: YCSB-C is 100% reads; use Zipf α=0.99 over [0..num_ops-1]
  rand_val(1);                 // seed the LCG
  const double alpha = 0.99;   // YCSB default skew
  std::vector<uint64_t> read_idx(num_ops);
  for (uint64_t i = 0; i < num_ops; ++i) {
    int rv = zipf(alpha, (int)num_ops);   // 1..num_ops
    uint64_t idx = (uint64_t)(rv - 1);    // 0..num_ops-1
    read_idx[i] = idx;
  }

  auto t_get0 = TIME_NOW;
#pragma omp parallel for num_threads(nthreads) schedule(static)
  for (uint64_t i = 0; i < num_ops; ++i) {
    const std::string& key = keys[read_idx[i]];
    // 1 MiB scratch buffer for lookup result
    std::unique_ptr<uint8_t[]> buf(new uint8_t[1 << 20]);
    std::span<const std::byte> k(reinterpret_cast<const std::byte*>(key.data()), key.size());
    table.lookup(k, buf.get());
  }
  auto get_time_us = std::chrono::duration_cast<std::chrono::microseconds>(TIME_NOW - t_get0).count();

  std::cout << "read_time: " << (get_time_us / 1e6) << " s\n";
  return 0;
}
