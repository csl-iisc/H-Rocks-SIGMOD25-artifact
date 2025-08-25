#include <rocksdb/sst_file_writer.h>
#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <iostream>
#include <omp.h>
#include <string>
#include <algorithm>
#include "bits/stdc++.h"
#include "batch.h"
#include "block_cache.cuh"

#define TOMBSTONE_MARKER NULL

using namespace ROCKSDB_NAMESPACE; 
using namespace std; 
#define TIME_NOW std::chrono::high_resolution_clock::now()    

int sstWriter(char *keys, char** values, uint64_t nkeys, uint32_t keyLen, uint32_t valueLen, int NTHREADS, rocksdb::DB *db, BlockCache *bCache, BCache *cache) 
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

    NTHREADS = 32;
    std::vector<std::string> file_path(NTHREADS); 
    for (int i = 0; i < NTHREADS; ++i) {
        SstFileWriter temp(EnvOptions(), options, options.comparator);
        sst_file_writer.push_back(new SstFileWriter(EnvOptions(), options, options.comparator)); 
        std::string i_str = std::to_string(i); 
        //file_path[i] = "/pmem/file" + i_str + ".sst";
        file_path[i] = "/dev/shm/file" + i_str + ".sst";
        Status s = sst_file_writer[i]->Open(file_path[i]);
        if (!s.ok()) {
            printf("Error while opening file %s, Error: %s\n", file_path[i].c_str(),
                    s.ToString().c_str());
            return 1;
        }
    }

    uint64_t num_elems_per_thread = nkeys/NTHREADS; 
    IngestExternalFileOptions ifo; 
    auto start = TIME_NOW; 
#pragma omp parallel for num_threads(NTHREADS)
    for(int j = 0; j < NTHREADS; j++) {
        std::string key, key_next; 
        for (uint64_t i = j * num_elems_per_thread; i < (j + 1) * num_elems_per_thread; i++) {
            if (i > nkeys - 1) 
                break; 
            key.assign(keys + i * keyLen, keys + (i + 1) * keyLen); 
            //bCache->invalidate(cache, keys + i * keyLen); 
            if (values[i] == TOMBSTONE_MARKER) 
                continue; 
            if(i + 2 < nkeys - 1) {
                key_next.assign(keys + (i + 1) * keyLen, keys + (i + 2) * keyLen); 
                if(key.compare(key_next) == 0) 
                    continue;
            }
#ifdef __PRINT_DEBUG__
            std::cout << "i: " << i << " key: " << key << " value: " << values[i] << "\n";  
#endif
            Status s = sst_file_writer[j]->Put(key, values[i]);
            if(!s.ok())
                cout << s.ToString() << "\n";
            assert(s.ok()); 
        }
        Status s = sst_file_writer[j]->Finish();
        //assert(s.ok()); 
        s = db->IngestExternalFile({file_path[j]}, ifo);
    }
    auto sst_file_insertion_time = (TIME_NOW - start).count(); 
    cout << "sst_insertion_time: " << sst_file_insertion_time/1000000.0 << "\n";

    cout << "INSERTED INTO FILE\n";

    return 0; 
}

    /*
int sstWriter(char *keys, char** values, uint64_t nkeys, uint32_t keyLen, uint32_t valueLen, int NTHREADS) 
{
    std::cout << "num keys: " << nkeys; 
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
    for (int i = 0; i < NTHREADS; ++i) {
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

    uint64_t num_elems_per_thread = nkeys/NTHREADS; 
    IngestExternalFileOptions ifo; 
    auto start = TIME_NOW; 
#pragma omp parallel for num_threads(NTHREADS)
    for(int j = 0; j < NTHREADS; j++) {
        for (uint64_t i = j * num_elems_per_thread; i < (j + 1) * num_elems_per_thread; ++i) {
            std::string key; 
            if (i > nkeys - 1) 
                break; 
            if (values[i] == TOMBSTONE_MARKER) 
                continue; 
            key.assign(keys + i * keyLen, keys + (i + 1) * keyLen); 
            //Slice skey = key; 
            //Slice svalue = values[i];
            std::cout << "i: " << i << " key: " << key << " value: " << values[i] << " "; 
            Status s = sst_file_writer[j]->Put(key, values[i]);
            if(!s.ok())
                cout << s.ToString() << "\n";
            assert(s.ok()); 
        }
        Status s = sst_file_writer[j]->Finish();
        assert(s.ok()); 
        s = db->IngestExternalFile({file_path[j]}, ifo);
    }

    auto sst_file_insertion_time = (TIME_NOW - start).count(); 
    cout << "INSERTED INTO FILE\n";
    */
    /*
       CompactionOptions coptions; 
       ifo.ingest_behind = true; 
       start = TIME_NOW; 
#pragma omp parallel for num_threads(NTHREADS/2)
for(int i = 0; i < NTHREADS/2; ++i) {
Status s = db->IngestExternalFile({file_path[i]}, ifo);
}
Status s = db->CompactFiles(coptions, file_path, 1); 
#pragma omp parallel for num_threads(NTHREADS/2)
for(int i = NTHREADS/2; i < NTHREADS; ++i) {
Status s = db->IngestExternalFile({file_path[i]}, ifo);
}
    auto ingestion_time = (TIME_NOW - start).count(); 
    std::cout << "sst_insertion_time (ms): " <<  sst_file_insertion_time/1000000.0 << "\n"; 
    //std::cout << "ingestion_time: " << ingestion_time/1000000.0 << "\n"; 
    sst_file_writer.clear(); 
    return 0;

    }

*/
