#include <thread>
#include <vector>
#include <rocksdb/sst_file_writer.h>
#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include "batch.h"
#include <iostream>
#include <omp.h>
#include <string>
#include <algorithm>
#include "bits/stdc++.h"

using namespace ROCKSDB_NAMESPACE; 

void process_chunk(uint32_t thread_id, uint64_t start_idx, uint64_t end_idx, char* keys, char** values, uint64_t nkeys, uint32_t keyLen, rocksdb::WritableFile** sst_file_writer) {
    for (uint64_t i = start_idx; i < end_idx; ++i) {
        std::string key;
        if (i > nkeys - 1)
            break;
        if (values[i] == TOMBSTONE_MARKER)
            continue;
        key.assign(keys + i * keyLen, keys + (i + 1) * keyLen);
        Status s = sst_file_writer[thread_id]->Put(key, values[i]);
        // error handling and assertions
        //     
    }
    Status s = sst_file_writer[thread_id]->Finish();
    // error handling and assertions
    // 
}

int sstWriter(uint64_t nkeys, int num_threads, rocksdb::DB *db) 
{
    Options options;
    options.num_levels = 1;
    options.compaction_style = kCompactionStyleNone;
    options.allow_ingest_behind = true; 
    options.write_buffer_size = 1024 * 1024 * 1024; 
    options.min_write_buffer_number_to_merge = 10; 
    options.level0_file_num_compaction_trigger = 10; 
    std::vector<SstFileWriter*> sst_file_writer; 
    // Path to where we will write the SST file
    
    std::vector<std::string> file_path(NTHREADS); 
        for (uint32_t i = 0; i < NTHREADS; ++i) 
        {
            SstFileWriter temp(EnvOptions(), options, options.comparator);
            sst_file_writer.push_back(new SstFileWriter(EnvOptions(), options, options.comparator)); 
            std::string i_str = std::to_string(i); 
            file_path[i] = "/dev/shm/file" + i_str + ".sst";
            Status s = sst_file_writer[i]->Open(file_path[i]);
            if (!s.ok()) {
                printf("Error while opening file %s, Error: %s\n", file_path[i].c_str(),
                        s.ToString().c_str());
                return 1;
            }
        }
        
    std::vector<std::thread> threads;
    uint64_t num_elems_per_thread = (nkeys + num_threads - 1) / num_threads;
    char** values_per_thread = new char*[num_threads];
    for (uint32_t i = 0; i < num_threads; ++i) {
        uint64_t start_idx = i * num_elems_per_thread;
        uint64_t end_idx = std::min((i + 1) * num_elems_per_thread, nkeys);
        values_per_thread[i] = values + start_idx;
        threads.emplace_back(process_chunk, i, start_idx, end_idx, keys, values_per_thread[i], nkeys, keyLen, sst_file_writer);

    }
    for (auto& thread : threads) {
        thread.join();

    }
    delete[] values_per_thread;

    std::vector<std::thread> threads2;
    uint32_t num_elems_per_thread2 = (num_threads + 1) / 2;
    for (uint32_t i = 0; i < 2; ++i) {
        uint32_t start_idx = i * num_elems_per_thread2;
        uint32_t end_idx = std::min((i + 1) * num_elems_per_thread2, num_threads);
        threads2.emplace_back(process_chunk, i, start_idx, end_idx, file_path, options, env_options);

    }
    for (auto& thread : threads2) {
        thread.join();

    }
    rocksdb::Status s = db->CompactFiles(coptions, file_path, 1);
    if (!s.ok()) {
        std::cerr << "Unable to compact files: " << s.ToString() << std::endl;

    }
    return 0;

}

