#include "batch.h"
#include "pmem_paths.h"
#include <cassert>
#include <cstdio>
#include <string>
#include <vector>
#include <functional>
#include <chrono>
#include <random>
#include <getopt.h>
#include <unistd.h>
#include <algorithm>
#include <rocksdb/db.h>
#include <rocksdb/slice.h>
#include <rocksdb/options.h>
#include <rocksdb/write_batch.h>
#include "rocksdb/statistics.h"

using namespace std;
using namespace rocksdb;

#define TRUE 1
#define FALSE 0

// ------- RNG + Zipf (same style as your B) -------
double rand_val(int seed) {
  const long  a = 16807;
  const long  m = 2147483647;
  const long  q = 127773;
  const long  r = 2836;
  static long x;
  long x_div_q, x_mod_q, x_new;
  if (seed > 0) { x = seed; return 0.0; }
  x_div_q = x / q;
  x_mod_q = x % q;
  x_new = (a * x_mod_q) - (r * x_div_q);
  x = (x_new > 0) ? x_new : x_new + m;
  return ((double)x / m);
}

int zipf(double alpha, int n) {
  static int first = TRUE;
  static double c = 0;
  static double *sum_probs;
  double z;
  int zipf_value;
  int i, low, high, mid;

  if (first == TRUE) {
    for (i = 1; i <= n; i++) c += (1.0 / pow((double)i, alpha));
    c = 1.0 / c;

    sum_probs = (double*)malloc((n + 1) * sizeof(*sum_probs));
    sum_probs[0] = 0;
    for (i = 1; i <= n; i++) sum_probs[i] = sum_probs[i - 1] + c / pow((double)i, alpha);
    first = FALSE;
  }

  do { z = rand_val(0); } while ((z == 0) || (z == 1));

  low = 1; high = n;
  do {
    mid = (int)floor((low + high) / 2.0);
    if (sum_probs[mid] >= z && sum_probs[mid - 1] < z) { zipf_value = mid; break; }
    else if (sum_probs[mid] >= z) { high = mid - 1; }
    else { low = mid + 1; }
  } while (low <= high);

  assert(zipf_value >= 1 && zipf_value <= n);
  return zipf_value;
}

// ------- key/value generation -------
typedef std::vector<char> char_array;
char_array charset() {
  return char_array({'A','B','C','D','E','F',
                     'G','H','I','J','K',
                     'L','M','N','O','P',
                     'Q','R','S','T','U',
                     'V','W','X','Y','Z'});
}

std::string generate_random_string(size_t length, std::function<char(void)> rand_char) {
  std::string str(length, 0);
  std::generate_n(str.begin(), length, rand_char);
  return str;
}

int main(int argc, char **argv) {
  int option_char;
  uint64_t num_ops = 0;
  size_t key_size = 0, value_size = 0;

  while ((option_char = getopt(argc, argv, ":n:k:v:")) != -1) {
    switch (option_char) {
      case 'n': num_ops   = atoi(optarg); break;
      case 'k': key_size  = atoi(optarg); break;
      case 'v': value_size= atoi(optarg); break;
      case ':': fprintf(stderr, "option needs a value\n"); return 2;
      case '?': fprintf(stderr, "usage: %s [-n ops] [-k key_size] [-v value_size]\n", argv[0]); return 2;
    }
  }

  // Mix: 90% reads (latest-biased), 10% inserts
  uint64_t num_puts = (uint64_t)(num_ops * 0.10);
  uint64_t num_gets = num_ops - num_puts;

  std::cout << "Number of puts (inserts): " << num_puts << std::endl;
  std::cout << "Number of gets (reads):   " << num_gets << std::endl;
  std::cout << "Key size: " << key_size << " | Value size: " << value_size << std::endl;

  // Open DB
  rocksdb::DB* db = nullptr;
  rocksdb::Options options;
  options.IncreaseParallelism(64);
  options.create_if_missing = true;
  rocksdb::Status s = rocksdb::DB::Open(options, hrocks::PmemPath("rdb_ycsbD"), &db);
  assert(s.ok());
  std::cout << "DB opened\n";

  // Prepare a Zipf(alpha) sampler over [1..num_puts] for "recency ranks"
  // alpha ~= 0.99 is YCSB default; use that here
  rand_val(1);                 // seed the LCG
  double alpha = 0.99;         // skew (community standard)
  std::vector<uint64_t> key_idx(num_gets);
  for (uint64_t i = 0; i < num_gets; i++) {
    int rank = zipf(alpha, (int)std::max<uint64_t>(1, num_puts));
    // Latest-biased: rank=1 → newest; map to index in [0..num_puts-1]
    // If keys are inserted in order 0..num_puts-1, newest is num_puts-1.
    uint64_t latest_index = (num_puts == 0) ? 0 : (num_puts - (uint64_t)rank);
    key_idx[i] = latest_index; // 0-based index into keys[]
  }

  // Generate keys/values for the INSERTS we will perform
  std::vector<std::string> keys(num_puts);
  std::vector<std::string> values(num_puts);
  const auto ch_set = charset();
  std::default_random_engine rng(std::random_device{}());
  std::uniform_int_distribution<> dist(0, (int)ch_set.size() - 1);
  auto randchar = [ch_set, &dist, &rng]() { return ch_set[dist(rng)]; };

  for (uint64_t i = 0; i < num_puts; ++i) {
    keys[i]   = generate_random_string(key_size   ? key_size - 1   : 0, randchar);
    values[i] = generate_random_string(value_size ? value_size - 1 : 0, randchar);
  }

  // Schedule operations with H-Rocks Batch
  std::vector<Command> readCommands, writeCommands, updateCommands;
  Batch batch(readCommands, writeCommands, updateCommands, 0, db);

  // First, perform the INSERTS (10% ops). These create the "latest" items.
  char *key = (char*)malloc(key_size ? key_size : 1);
  char *value = (char*)malloc(value_size ? value_size : 1);

  for (uint64_t i = 0; i < num_puts; ++i) {
    strcpy(key,   keys[i].c_str());
    strcpy(value, values[i].c_str());
    batch.Put(key, value);
  }

  // Then, perform READS (90%) with latest bias over the inserted set
  for (uint64_t i = 0; i < num_gets; ++i) {
    uint64_t idx = (num_puts == 0) ? 0 : std::min<uint64_t>(key_idx[i], num_puts - 1);
    strcpy(key, keys[idx].c_str());
    batch.Get(key);
  }

  batch.Exit();

  free(key);
  free(value);
  return 0;
}
