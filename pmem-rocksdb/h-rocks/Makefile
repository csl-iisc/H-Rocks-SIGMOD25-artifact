# --- H-Rocks (hrocks) Makefile (cleaned) -------------------------------------
# Usage:
#   make            # builds lib + all tests
#   make lib        # builds libgpu.a
#   make tests      # builds all test binaries
#   make clean
#
# Customize:
#   make SM=90      # CUDA arch (default: 75)
#   make DEBUG=1    # add debug flags
# -----------------------------------------------------------------------------

# include ../make_config.mk

ifndef DISABLE_JEMALLOC
  ifdef JEMALLOC
    PLATFORM_CXXFLAGS += -DROCKSDB_JEMALLOC -DJEMALLOC_NO_DEMANGLE
  endif
  EXEC_LDFLAGS := $(JEMALLOC_LIB) $(EXEC_LDFLAGS) -lpthread
  PLATFORM_CXXFLAGS += $(JEMALLOC_INCLUDE)
endif

# RTTI is needed by RocksDB tools & some tests; keep default enabled
ifneq ($(USE_RTTI), 1)
  # CXXFLAGS += -fno-rtti
endif

include ../dcpmm.mk
EXEC_LDFLAGS += $(LDFLAGS)

# -------------------------- Toolchain & Paths --------------------------------
CXX        ?= g++
NVCC       ?= nvcc
BIN_DIR    ?= bin
LIB_NAME   ?= libgpu.a
SM         ?= 75

INCLUDES := -I. -I$(BIN_DIR) -I../libgpm/include -I/usr/local/cuda/include -I../include

# ------------------------------ Flags ----------------------------------------
HOST_CXXFLAGS := -O3 -std=c++14 $(PLATFORM_CXXFLAGS)
CUDA_CXXFLAGS := -O3 -std=c++14 -Xcompiler -fopenmp -rdc=true -arch=sm_$(SM)

# Link libs (CUDA + system)
EXEC_LDFLAGS+= -lzstd -lsnappy -llz4 -lbz2 -lz
CUDA_LDLIBS := -L/usr/local/cuda/lib64 -lcuda -lcudart
SYS_LDLIBS  := -lpthread -lrt -lm
PMEM_LDLIBS ?= -lpmem
NVIDIA_ML   ?= -lnvidia-ml
PCI_LIB     ?= -lpci

# Debug knobs
ifdef DEBUG
  CUDA_CXXFLAGS += -g -G
  HOST_CXXFLAGS += -g # (enable ASan if you want) -fsanitize=address
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
  test_ycsb \
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
	$(NVCC) $(CUDA_CXXFLAGS) $(INCLUDES) \
	  $(CUDA_LDLIBS) $(SYS_LDLIBS) $(PMEM_LDLIBS) $(NVIDIA_ML) $(PCI_LIB) \
	  $(LIB_NAME) $< -o $@

$(BIN_DIR)/test: test.cc $(LIB_NAME) | $(BIN_DIR)
	$(NVCC) $(CUDA_CXXFLAGS) $(INCLUDES) \
	  $(CUDA_LDLIBS) $(SYS_LDLIBS) $(PMEM_LDLIBS) $(NVIDIA_ML) $(PCI_LIB) \
	  $(LIB_NAME) $< -o $@

$(BIN_DIR)/memcpy_experiment: memcpy_experiment.cu | $(BIN_DIR)
	$(NVCC) $(CUDA_CXXFLAGS) $(INCLUDES) \
	  $(CUDA_LDLIBS) $(SYS_LDLIBS) $(PMEM_LDLIBS) \
	  $< -o $@

# Tests that link against RocksDB and platform flags
define LINK_ROCKSDB_TEMPLATE
$(BIN_DIR)/$(1): $(1).cc $(LIB_NAME) | $(BIN_DIR)
	$$(NVCC) $$(CUDA_CXXFLAGS) $$(INCLUDES) \
	  $$(LIB_NAME) ../librocksdb.a \
	  $$(PLATFORM_LDFLAGS) $$(EXEC_LDFLAGS) \
	  $$(CUDA_LDLIBS) $$(SYS_LDLIBS) $$(PMEM_LDLIBS) $$(NVIDIA_ML) $$(PCI_LIB) \
	  -o $$@ $$<
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