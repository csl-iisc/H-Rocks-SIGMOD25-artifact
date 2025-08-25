set(CMAKE_CXX_COMPILER "/usr/bin/x86_64-linux-gnu-g++-11")

cmake_minimum_required(VERSION 3.16)
project(Plush)
include(ExternalProject)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native -pthread -gdwarf-4")


set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
#find_package(OpenMP REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()



add_library(hashtable SHARED
        src/hashtable/Hashtable.h
        src/hashtable/Hashtable.cpp
        include/pibench/tree_api.h
        src/benchmarking/PibenchWrapper.cpp
        src/benchmarking/PibenchWrapper.h)

add_library(hashtable_var SHARED
        src/hashtable/Hashtable.h
        src/hashtable/Hashtable.cpp
        include/pibench/tree_api.h
        src/benchmarking/PibenchWrapperVar.cpp
        src/benchmarking/PibenchWrapperVar.h)


add_library(utree SHARED
        include/pibench/tree_api.h
        include/utree/PibenchWrapper.cpp
        include/utree/PibenchWrapper.h)
target_link_libraries(utree /usr/local/lib/libpmemobj.so /usr/local/lib/libpmem.so )


add_executable(gutenberg
        examples/gutenberg.cpp
        src/hashtable/Hashtable.h
        )


add_executable(demo
        src/hashtable/Hashtable.h
        examples/example.cpp
        )


add_executable(test_put
        examples/test_put.cpp
        src/hashtable/Hashtable.h
        )

add_executable(test_get
        examples/test_get.cpp
        src/hashtable/Hashtable.h
        )

add_executable(test_put_mt
        examples/test_put_mt.cpp
        src/hashtable/Hashtable.h
        )

add_executable(test_get_mt
        examples/test_get_mt.cpp
        src/hashtable/Hashtable.h
        )


add_executable(test_update
        examples/test_update.cpp
        src/hashtable/Hashtable.h
        )

add_executable(test_put_int
        examples/test_put_int.cpp
        src/hashtable/Hashtable.h
        )

add_executable(test_get_int
        examples/test_get_int.cpp
        src/hashtable/Hashtable.h
        )

add_executable(test_ycsbA
        examples/test_ycsbA.cpp
        src/hashtable/Hashtable.h
        )

add_executable(test_ycsbB
        examples/test_ycsbB.cpp
        src/hashtable/Hashtable.h
        )



target_link_libraries(gutenberg PRIVATE hashtable)
target_link_libraries(demo PRIVATE hashtable)
target_link_libraries(test_put PRIVATE hashtable)
target_link_libraries(test_get PRIVATE hashtable)
target_link_libraries(test_put_mt PRIVATE hashtable)
target_link_libraries(test_get_mt PRIVATE hashtable)
target_link_libraries(test_update PRIVATE hashtable)
target_link_libraries(test_put_int PRIVATE hashtable)
target_link_libraries(test_get_int PRIVATE hashtable)
target_link_libraries(test_ycsbA PRIVATE hashtable)
target_link_libraries(test_ycsbB PRIVATE hashtable)

add_subdirectory(test)

add_subdirectory(dash)
add_subdirectory(viper)
add_subdirectory(pmemkv)
add_subdirectory(dptree)
add_subdirectory(fptree)
add_subdirectory(pibench)
add_subdirectory(faster/cc)

#missing rocksdb

# FASTFAIR is make-based
#ExternalProject_Add(fast_fair
#        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/fast_fair/concurrent_pmdk/
#        BUILD_IN_SOURCE true
#        CONFIGURE_COMMAND ""
#        BUILD_COMMAND make fastfairlib
#        INSTALL_COMMAND ""
#        )
