include ../../make_config.mk

ifndef DISABLE_JEMALLOC
	ifdef JEMALLOC
		PLATFORM_CXXFLAGS += -DROCKSDB_JEMALLOC -DJEMALLOC_NO_DEMANGLE
	endif
	EXEC_LDFLAGS := $(JEMALLOC_LIB) $(EXEC_LDFLAGS) -lpthread
	PLATFORM_CXXFLAGS += $(JEMALLOC_INCLUDE)
endif

ifneq ($(USE_RTTI), 1)
	#CXXFLAGS += -fno-rtti
endif

include ../../dcpmm.mk
EXEC_LDFLAGS += $(LDFLAGS)

PLATFORM_CXXFLAGS=-std=c++14 -DHAVE_ALIGNED_NEW -DROCKSDB_PLATFORM_POSIX -DROCKSDB_LIB_IO_POSIX  -DOS_LINUX -DROCKSDB_FALLOCATE_PRESENT -DSNAPPY -DGFLAGS=1 -DZLIB -DBZIP2 -DNUMA -DTBB -DROCKSDB_MALLOC_USABLE_SIZE -DROCKSDB_PTHREAD_ADAPTIVE_MUTEX -DROCKSDB_BACKTRACE -DROCKSDB_RANGESYNC_PRESENT -DROCKSDB_SCHED_GETCPU_PRESENT -DROCKSDB_AUXV_GETAUXVAL_PRESENT -DHAVE_SSE42  -DHAVE_PCLMUL  -DHAVE_AVX2 -DHAVE_UINT128_EXTENSION -DROCKSDB_SUPPORT_THREAD_LOCAL

PLATFORM_CXXFLAGS2=-std=c++14 -DHAVE_ALIGNED_NEW -DROCKSDB_PLATFORM_POSIX -DROCKSDB_LIB_IO_POSIX  -DOS_LINUX -DROCKSDB_FALLOCATE_PRESENT -DSNAPPY -DGFLAGS=1 -DZLIB -DBZIP2 -DNUMA -DTBB -DROCKSDB_MALLOC_USABLE_SIZE -DROCKSDB_PTHREAD_ADAPTIVE_MUTEX -DROCKSDB_BACKTRACE -DROCKSDB_RANGESYNC_PRESENT -DROCKSDB_SCHED_GETCPU_PRESENT -DROCKSDB_AUXV_GETAUXVAL_PRESENT -DHAVE_SSE42  -DHAVE_PCLMUL  -DHAVE_AVX2 -DHAVE_UINT128_EXTENSION -DROCKSDB_SUPPORT_THREAD_LOCAL

CXX = g++
#CXXFLAGS = -std=c++14 -lpthread -O3
LIB_NAME = libgpu.a
BIN_DIR = bin
NVCC = nvcc
INCLUDE_FILE = -I. -I./bin -I../libgpm/include -I/usr/local/cuda/include -I../../include
NVCCFLAGS = -lpmem -rdc=true -lpthread -lnvidia-ml -arch=sm_75 -lpci -Xcompiler -fopenmp -lrt -lm -O3

#DEBUG := 1 

ifdef DEBUG 
	NVCCFLAGS += -g -G 
	CXXFFLAGS += -g -fsanitize=address
endif 


all: $(BIN_DIR)/main $(BIN_DIR)/test

lib: $(BIN_DIR)/gpu_puts.o $(BIN_DIR)/gpu_gets.o $(BIN_DIR)/memtable.o $(BIN_DIR)/gpu_updates.o $(BIN_DIR)/search.o $(BIN_DIR)/batch.o $(BIN_DIR)/block_cache.o $(BIN_DIR)/sst_writer.o $(BIN_DIR)/gpu_range.o $(BIN_DIR)/gpu_puts_values.o $(BIN_DIR)/gpu_gets_values.o
	ar rcs $(LIB_NAME) $(BIN_DIR)/gpu_puts.o $(BIN_DIR)/gpu_gets.o $(BIN_DIR)/memtable.o $(BIN_DIR)/gpu_updates.o $(BIN_DIR)/search.o $(BIN_DIR)/batch.o $(BIN_DIR)/block_cache.o $(BIN_DIR)/sst_writer.o $(BIN_DIR)/gpu_range.o $(BIN_DIR)/gpu_puts_values.o $(BIN_DIR)/gpu_gets_values.o

#lib: $(BIN_DIR)/sender.o $(BIN_DIR)/batch.o  $(BIN_DIR)/gpu_puts.o
#	ar rcs $(LIB_NAME) $(BIN_DIR)/sender.o $(BIN_DIR)/receiver.o  $(BIN_DIR)/gpu_puts.o

$(BIN_DIR)/sender.o: sender.cpp sender.h command.h
	mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -c sender.cpp -o $(BIN_DIR)/sender.o

#$(BIN_DIR)/receiver.o: receiver.cpp receiver.h command.h
#	mkdir -p $(BIN_DIR)
#	$(CXX) $(CXXFLAGS) -c receiver.cpp -o $(BIN_DIR)/receiver.o

$(BIN_DIR)/gpu_puts.o: gpu_puts.cu gpu_puts.cuh 
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FILE) -c gpu_puts.cu -o $(BIN_DIR)/gpu_puts.o 

$(BIN_DIR)/gpu_puts_values.o: gpu_puts_values.cu gpu_puts_values.cuh 
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FILE) -c gpu_puts_values.cu -o $(BIN_DIR)/gpu_puts_values.o 

$(BIN_DIR)/gpu_gets_values.o: gpu_gets_values.cu gpu_gets_values.cuh 
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FILE) -c gpu_gets_values.cu -o $(BIN_DIR)/gpu_gets_values.o 

$(BIN_DIR)/memtable.o: memtable.cu memtable.cuh 
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FILE) -c memtable.cu -o $(BIN_DIR)/memtable.o 

$(BIN_DIR)/gpu_gets.o: gpu_gets.cu gpu_gets.cuh 
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -c  $(INCLUDE_FILE) gpu_gets.cu -o $(BIN_DIR)/gpu_gets.o

$(BIN_DIR)/search.o: search.cu search.cuh 
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -c  $(INCLUDE_FILE) search.cu -o $(BIN_DIR)/search.o

$(BIN_DIR)/gpu_updates.o: gpu_updates.cu gpu_updates.cuh 
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FILE) -c gpu_updates.cu -o $(BIN_DIR)/gpu_updates.o

$(BIN_DIR)/batch.o: batch.h batch.cu 
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(PLATFORM_LDFLAGS) -I../../include $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS) -c batch.cu -o $(BIN_DIR)/batch.o $(INCLUDE_FILE) 

$(BIN_DIR)/sst_writer.o: sst_writer.h sst_writer.cc 
	mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -c sst_writer.cc -o $(BIN_DIR)/sst_writer.o -I../../include -O3 -std=c++11 $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS) -fopenmp

$(BIN_DIR)/block_cache.o: block_cache.cuh block_cache.cu
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -c block_cache.cu -o $(BIN_DIR)/block_cache.o $(INCLUDE_FILE)

$(BIN_DIR)/gpu_range.o: gpu_range.cu gpu_range.cuh 
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FILE) -c gpu_range.cu -o $(BIN_DIR)/gpu_range.o 

$(BIN_DIR)/main:
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -L/usr/local/cuda/lib64 -lcuda -lcudart $(LIB_NAME) main.cc -o ./bin/main

$(BIN_DIR)/test:
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -L/usr/local/cuda/lib64 -lcuda -lcudart $(LIB_NAME) test.cc -o ./bin/test

$(BIN_DIR)/test_get_put:
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(INCLUDE_FILE) $(LIB_NAME) ../../librocksdb.a $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS) -std=c++14 -L/usr/local/cuda/lib64 -lcuda -lcudart test_get_put.cc -o ./bin/test_get_put 

$(BIN_DIR)/test_puts: test_puts.cc
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(INCLUDE_FILE) $(LIB_NAME) ../../librocksdb.a $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS) -std=c++14 -L/usr/local/cuda/lib64 -lcuda -lcudart test_puts.cc -o ./bin/test_puts 

$(BIN_DIR)/test_updates:
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(INCLUDE_FILE) $(LIB_NAME) ../../librocksdb.a $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS) -std=c++14 -L/usr/local/cuda/lib64 -lcuda -lcudart test_updates.cc -o ./bin/test_updates

$(BIN_DIR)/test_ycsbA:
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(INCLUDE_FILE) $(LIB_NAME) ../../librocksdb.a $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS) -std=c++14 -L/usr/local/cuda/lib64 -lcuda -lcudart test_ycsbA.cc -o ./bin/test_ycsbA

$(BIN_DIR)/test_ycsbB:
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(INCLUDE_FILE) $(LIB_NAME) ../../librocksdb.a $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS) -std=c++14 -L/usr/local/cuda/lib64 -lcuda -lcudart test_ycsbB.cc -o ./bin/test_ycsbB

$(BIN_DIR)/test_ycsbC:
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(INCLUDE_FILE) $(LIB_NAME) ../../librocksdb.a $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS) -std=c++14 -L/usr/local/cuda/lib64 -lcuda -lcudart test_ycsbC.cc -o ./bin/test_ycsbC

$(BIN_DIR)/test_ycsb:
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(INCLUDE_FILE) $(LIB_NAME) ../../librocksdb.a $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS) -std=c++14 -L/usr/local/cuda/lib64 -lcuda -lcudart test_ycsb.cc -o ./bin/test_ycsb

$(BIN_DIR)/test_atomic_txn:
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(INCLUDE_FILE) $(LIB_NAME) ../../librocksdb.a $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS) -std=c++14 -L/usr/local/cuda/lib64 -lcuda -lcudart test_atomic_txn.cc -o ./bin/test_atomic_txn

$(BIN_DIR)/test_batch:
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(INCLUDE_FILE) $(LIB_NAME) ../../librocksdb.a $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS) -std=c++14 -L/usr/local/cuda/lib64 -lcuda -lcudart test_batch.cc -o ./bin/test_batch


$(BIN_DIR)/test_range:
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(INCLUDE_FILE) $(LIB_NAME) ../../librocksdb.a $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS) -std=c++14 -L/usr/local/cuda/lib64 -lcuda -lcudart test_range.cc -o ./bin/test_range

$(BIN_DIR)/uniform_dist_req_gen:
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(INCLUDE_FILE) $(LIB_NAME) ../../librocksdb.a $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS) -std=c++14 -L/usr/local/cuda/lib64 -lcuda -lcudart uniform_dist_req_gen.cc -o ./bin/uniform_dist_req_gen

$(BIN_DIR)/test_same_key:
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(INCLUDE_FILE) $(LIB_NAME) ../../librocksdb.a $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS) -std=c++14 -L/usr/local/cuda/lib64 -lcuda -lcudart test_same_key.cc -o ./bin/test_same_key

$(BIN_DIR)/test_multiple_batches:
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(INCLUDE_FILE) $(LIB_NAME) ../../librocksdb.a $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS) -std=c++14 -L/usr/local/cuda/lib64 -lcuda -lcudart test_multiple_batches.cc -o ./bin/test_multiple_batches


$(BIN_DIR)/memcpy_experiment:
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FILE) -std=c++14 -L/usr/local/cuda/lib64 -lcuda -lcudart memcpy_experiment.cu -o ./bin/memcpy_experiment

clean:
	rm -rf $(BIN_DIR) $(LIB_NAME)

