#include <rocksdb/sst_file_writer.h>
#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/env.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "batch.h"
#include "block_cache.cuh"

#define TOMBSTONE_MARKER nullptr
#define TIME_NOW std::chrono::high_resolution_clock::now()

using namespace ROCKSDB_NAMESPACE;

static inline uint64_t div_up(uint64_t x, uint64_t y) { return (x + y - 1) / y; }

int sstWriter(char* keys,
              char** values,
              uint64_t nkeys,
              uint32_t keyLen,
              uint32_t valueLen,
              int NTHREADS,
              rocksdb::DB* db,
              BlockCache* /*bCache*/,
              BCache* /*cache*/) {
  // ---- Options ----
  rocksdb::Options options;
  options.comparator = rocksdb::BytewiseComparator();   // make comparator explicit
  options.num_levels = 1;
  options.compaction_style = rocksdb::kCompactionStyleNone;
  options.allow_ingest_behind = true;
  options.write_buffer_size = 1024ull * 1024ull * 1024ull;
  options.min_write_buffer_number_to_merge = 10;
  options.level0_file_num_compaction_trigger = 10;

  rocksdb::EnvOptions env_opts;

  // Clamp threads for small batches
  NTHREADS = std::max(1, std::min<int>(NTHREADS, static_cast<int>(std::max<uint64_t>(1, nkeys))));
  const uint64_t chunk = div_up(nkeys, static_cast<uint64_t>(NTHREADS));

  std::vector<std::unique_ptr<rocksdb::SstFileWriter>> writers;
  std::vector<std::string> file_path(NTHREADS);
  std::vector<uint64_t> written_counts(NTHREADS, 0);
  writers.reserve(NTHREADS);

  for (int i = 0; i < NTHREADS; ++i) {
    file_path[i] = std::string("/dev/shm/file") + std::to_string(i) + ".sst";
    std::unique_ptr<rocksdb::SstFileWriter> w(new rocksdb::SstFileWriter(env_opts, options));
    rocksdb::Status s = w->Open(file_path[i]);
    if (!s.ok()) {
      std::cerr << "Open(" << file_path[i] << ") failed: " << s.ToString() << "\n";
      return 1;
    }
    writers.emplace_back(std::move(w));
  }

  rocksdb::IngestExternalFileOptions ifo;
  auto start = TIME_NOW;

#pragma omp parallel for num_threads(NTHREADS) schedule(static)
  for (int t = 0; t < NTHREADS; ++t) {
    const uint64_t begin = static_cast<uint64_t>(t) * chunk;
    const uint64_t end   = std::min(begin + chunk, nkeys);
    uint64_t local_writes = 0;

    if (begin >= end) {
      // still finalize an empty file
      (void)writers[t]->Finish();
      continue;
    }

    std::string key;
    key.resize(keyLen);

    for (uint64_t i = begin; i < end; ++i) {
      if (values[i] == TOMBSTONE_MARKER) continue;

      // exact keyLen bytes
      std::memcpy(&key[0], keys + i * keyLen, keyLen);

      // optional dedup within thread’s local range
      if (i + 1 < end) {
        if (std::memcmp(keys + i * keyLen, keys + (i + 1) * keyLen, keyLen) == 0) continue;
      }

      // values[i] is a pointer to valueLen bytes (may contain zeros)
      rocksdb::Status s = writers[t]->Put(rocksdb::Slice(key),
                                          rocksdb::Slice(values[i], valueLen));
      if (!s.ok()) {
        fprintf(stderr, "[thread %d] Put failed at i=%lu: %s\n",
                t, (unsigned long)i, s.ToString().c_str());
      }
      ++local_writes;
    }

    rocksdb::Status s = writers[t]->Finish();
    if (!s.ok()) {
      fprintf(stderr, "[thread %d] Finish failed: %s\n", t, s.ToString().c_str());
    }
    written_counts[t] = local_writes;
  }

  // Ingest files sequentially (safe & simple)
  for (int t = 0; t < NTHREADS; ++t) {
    if (written_counts[t] == 0) {
      rocksdb::Status del_status = rocksdb::Env::Default()->DeleteFile(file_path[t]);
      if (!del_status.ok()) {
        std::cerr << "DeleteFile(" << file_path[t] << ") failed: " << del_status.ToString() << "\n";
      }
      continue;
    }
    rocksdb::Status s = db->IngestExternalFile({file_path[t]}, ifo);
    if (!s.ok()) {
      std::cerr << "IngestExternalFile(" << file_path[t] << ") failed: " << s.ToString() << "\n";
      return 1;
    }
  }

  auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(TIME_NOW - start).count();
  std::cout << "sst_insertion_time: " << (ns / 1e6) << "\n";
  std::cout << "INSERTED INTO FILE\n";
  return 0;
}
