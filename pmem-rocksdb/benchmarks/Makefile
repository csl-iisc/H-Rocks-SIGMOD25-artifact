include ../make_config.mk

BIN_DIR ?= bin

ifndef DISABLE_JEMALLOC
	ifdef JEMALLOC
		PLATFORM_CXXFLAGS += -DROCKSDB_JEMALLOC -DJEMALLOC_NO_DEMANGLE
	endif
	EXEC_LDFLAGS := $(JEMALLOC_LIB) $(EXEC_LDFLAGS) -lpthread
	PLATFORM_CXXFLAGS += $(JEMALLOC_INCLUDE)
endif

ifneq ($(USE_RTTI), 1)
	CXXFLAGS += -fno-rtti
endif

include ../dcpmm.mk
EXEC_LDFLAGS += $(LDFLAGS)

.PHONY: clean librocksdb begin

all: begin $(BIN_DIR)/test_puts $(BIN_DIR)/test_deletes $(BIN_DIR)/test_gets $(BIN_DIR)/test_updates \
	$(BIN_DIR)/test_ranges $(BIN_DIR)/test_ycsbA $(BIN_DIR)/test_ycsbB $(BIN_DIR)/test_ycsbC $(BIN_DIR)/test_ycsbD

begin:
	mkdir -p $(BIN_DIR)

bin/test_puts: librocksdb test_puts.cc
	$(CXX) $(CXXFLAGS) test_puts.cc -o$@ ../librocksdb.a -I../include -O2 -std=c++11 $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS)

bin/test_deletes: librocksdb test_deletes.cc
	$(CXX) $(CXXFLAGS) test_deletes.cc -o$@ ../librocksdb.a -I../include -O2 -std=c++11 $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS)

bin/test_gets: librocksdb test_gets.cc
	$(CXX) $(CXXFLAGS) test_gets.cc -o$@ ../librocksdb.a -I../include -O2 -std=c++11 $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS)

bin/test_updates: librocksdb test_updates.cc
	$(CXX) $(CXXFLAGS) test_updates.cc -o$@ ../librocksdb.a -I../include -O2 -std=c++11 $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS)

bin/test_ranges: librocksdb test_ranges.cc
	$(CXX) $(CXXFLAGS) test_ranges.cc -o$@ ../librocksdb.a -I../include -O2 -std=c++11 $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS)

bin/test_ycsbA: librocksdb test_ycsbA.cc
	$(CXX) $(CXXFLAGS) test_ycsbA.cc -o$@ ../librocksdb.a -I../include -O2 -std=c++11 $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS)

bin/test_ycsbB: librocksdb test_ycsbB.cc
	$(CXX) $(CXXFLAGS) test_ycsbB.cc -o$@ ../librocksdb.a -I../include -O2 -std=c++11 $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS)

bin/test_ycsbC: librocksdb test_ycsbC.cc
	$(CXX) $(CXXFLAGS) test_ycsbC.cc -o$@ ../librocksdb.a -I../include -O2 -std=c++11 $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS)

bin/test_ycsbD: librocksdb test_ycsbD.cc
	$(CXX) $(CXXFLAGS) test_ycsbD.cc -o$@ ../librocksdb.a -I../include -O2 -std=c++11 $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS)

clean:
	rm -rf $(BIN_DIR)
