# --- H-Rocks (hrocks) Makefile ----------------------------------------------
# Usage:
#   make              # builds lib + all tests
#   make lib          # builds libgpu.a
#   make tests        # builds all test binaries
#   make clean
#
# Customize:
#   make SM=86        # CUDA arch (default: 86; set 75 for T4/V100)
#   make DEBUG=1      # debug flags
# -----------------------------------------------------------------------------

include ../make_config.mk
include ../dcpmm.mk

# -------------------------- Toolchain & Paths --------------------------------
CXX        ?= g++
NVCC       ?= nvcc
BIN_DIR    ?= bin
LIB_NAME   ?= libgpu.a
SM         ?= 75

# Header search paths (this dir layout is for pmem-rocksdb/h-rocks/)
INCLUDES := -I/usr/local/include -I. -I$(BIN_DIR) -I../libgpm/include -I/usr/local/cuda/include -I../include

# ------------------------------ Flags ----------------------------------------
# These macros mirror your working line; keep them on compile & link
PLATFORM_CXXDEFS := -DON_DCPMM \
  -DHAVE_ALIGNED_NEW -DROCKSDB_PLATFORM_POSIX -DROCKSDB_LIB_IO_POSIX -DOS_LINUX \
  -DROCKSDB_FALLOCATE_PRESENT -DSNAPPY -DGFLAGS=1 -DZLIB -DBZIP2 -DNUMA -DTBB \
  -DROCKSDB_MALLOC_USABLE_SIZE -DROCKSDB_PTHREAD_ADAPTIVE_MUTEX -DROCKSDB_BACKTRACE \
  -DROCKSDB_RANGESYNC_PRESENT -DROCKSDB_SCHED_GETCPU_PRESENT -DROCKSDB_AUXV_GETAUXVAL_PRESENT \
  -DHAVE_SSE42 -DHAVE_PCLMUL -DHAVE_AVX2 -DHAVE_UINT128_EXTENSION -DROCKSDB_SUPPORT_THREAD_LOCAL

HOST_CXXFLAGS := -O3 -std=c++14 $(PLATFORM_CXXFLAGS) $(PLATFORM_CXXDEFS)
CUDA_CXXFLAGS := -O3 -std=c++14 -Xcompiler -fopenmp -rdc=true -arch=sm_$(SM) $(PLATFORM_CXXDEFS)

# Libraries / link search paths
LIB_DIRS    := -L/usr/local/lib/ -L/usr/local/lib64/ -L/usr/local/cuda/lib64
CODEC_LIBS  := -lsnappy -lgflags -lz -lbz2 -llz4 -lzstd
SYS_LDLIBS  := -ldl -lpthread -lrt -lm -lnuma -ltbb
PMEM_LIBS   := -lpmem -lpmemobj
CUDA_LDLIBS := -lcuda -lcudart
EXTRA_LIBS  := -lnvidia-ml -lpci

# Compose EXEC_LDFLAGS (order matters but we’ll still place libs after object)
EXEC_LDFLAGS := $(LIB_DIRS) $(CODEC_LIBS) $(SYS_LDLIBS) $(PMEM_LIBS) $(CUDA_LDLIBS) $(EXTRA_LIBS)
EXEC_LDFLAGS += $(PLATFORM_LDFLAGS) $(LDFLAGS)

# Debug knobs
ifdef DEBUG
  CUDA_CXXFLAGS += -g -G
  HOST_CXXFLAGS += -g
endif

# --------------------------- Sources & Objects -------------------------------
GPU_OBJS := \
  $(BIN_DIR)/gpu_puts.o \
  $(BIN_DIR)/gpu_puts_values.o \
  $(BIN_DIR)/gpu_gets_values.o \
  $(BIN_DIR)/memtable.o \
  $(BIN_DIR)/gpu_gets.o \
  $(BIN_DIR)/search.o \
  $(BIN_DIR)/gpu_updates.o \
  $(BIN_DIR)/batch.o \
  $(BIN_DIR)/block_cache.o \
  $(BIN_DIR)/sst_writer.o \
  $(BIN_DIR)/gpu_range.o

# Binaries that DO NOT need RocksDB at link time
APPS_SIMPLE := memcpy_experiment

# Binaries that DO need RocksDB at link time
TESTS_ROCKS := \
  test_get_put \
  test_puts \
  test_updates \
  test_ycsbA \
  test_ycsbB \
  test_ycsbC \
  test_ycsbD \
  test_atomic_txn \
  test_batch \
  test_range \
  uniform_dist_req_gen \
  test_same_key \
  test_multiple_batches

ALL_BINS := $(addprefix $(BIN_DIR)/,$(APPS_SIMPLE) $(TESTS_ROCKS))

# ------------------------------- Targets -------------------------------------
.PHONY: all lib tests apps clean help
all: lib tests

lib: $(LIB_NAME)

tests: $(ALL_BINS)
apps:  $(addprefix $(BIN_DIR)/,$(APPS_SIMPLE))

$(LIB_NAME): $(GPU_OBJS)
	@mkdir -p $(BIN_DIR)
	ar rcs $@ $^

# --------------------------- Compile Rules -----------------------------------
$(BIN_DIR)/%.o: %.cu | $(BIN_DIR)
	$(NVCC) $(CUDA_CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BIN_DIR)/%.o: %.cc | $(BIN_DIR)
	$(CXX)  $(HOST_CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# ----------------------------- Link Rules ------------------------------------
# Simple apps (no RocksDB at link time)
$(BIN_DIR)/main: main.cc $(LIB_NAME) | $(BIN_DIR)
	$(NVCC) $(CUDA_CXXFLAGS) $(INCLUDES) main.cc -o $@ \
	  $(LIB_NAME) $(EXEC_LDFLAGS)

$(BIN_DIR)/test: test.cc $(LIB_NAME) | $(BIN_DIR)
	$(NVCC) $(CUDA_CXXFLAGS) $(INCLUDES) test.cc -o $@ \
	  $(LIB_NAME) $(EXEC_LDFLAGS)

$(BIN_DIR)/memcpy_experiment: memcpy_experiment.cu | $(BIN_DIR)
	$(NVCC) $(CUDA_CXXFLAGS) $(INCLUDES) memcpy_experiment.cu -o $@ \
	  $(EXEC_LDFLAGS)

# Tests that link against RocksDB (ORDER: source -> -o -> libs)
define LINK_ROCKSDB_TEMPLATE
$(BIN_DIR)/$(1): $(1).cc $(LIB_NAME) | $(BIN_DIR)
	$$(NVCC) $$(CUDA_CXXFLAGS) $$(INCLUDES) $$(PLATFORM_CXXDEFS) $$< -o $$@ \
	  $$(LIB_NAME) ../librocksdb.a \
	  $$(EXEC_LDFLAGS)
endef
$(foreach t,$(TESTS_ROCKS),$(eval $(call LINK_ROCKSDB_TEMPLATE,$(t))))

# ------------------------------- Clean ---------------------------------------
clean:
	rm -rf $(BIN_DIR) $(LIB_NAME)

# ------------------------------- Help ----------------------------------------
help:
	@echo "Targets:"
	@echo "  make [SM=75] [DEBUG=1]   Build lib + all tests"
	@echo "  make lib                  Build libgpu.a"
	@echo "  make tests                Build all test binaries"
	@echo "  make apps                 Build simple apps (main,test,memcpy_experiment)"
	@echo "  make clean                Remove bin/ and libgpu.a"
