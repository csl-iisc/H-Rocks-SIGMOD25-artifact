cmake_minimum_required(VERSION 3.18)

project(test_viper)

include(FetchContent)

cmake_policy(SET CMP0077 NEW)

FetchContent_Declare(
    viper
    GIT_REPOSITORY https://github.com/hpides/viper.git
    )
FetchContent_MakeAvailable(viper)



FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG "origin/main"
    )
set(BENCHMARK_USE_BUNDLED_GTEST OFF)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
set(BENCHMARK_ENABLE_TESTING OFF)
FetchContent_MakeAvailable(benchmark)

# PMDK
set(PMDK_INCLUDE_HINTS ${VIPER_PMDK_PATH}/include)
set(PMDK_LIB_HINTS ${VIPER_PMDK_PATH}/lib)

find_path(PMDK_INCLUDE_DIRS libpmem.h HINTS ${PMDK_INCLUDE_HINTS})
find_library(PMDK_LIBRARIES NAMES pmem libpmem HINTS ${PMDK_LIB_HINTS})

set(PMDK_FOUND true)
if (NOT PMDK_INCLUDE_DIRS OR "${PMDK_INCLUDE_DIRS}" STREQUAL "")
    message(FATAL_ERROR "libpmem include directory not found in default locations or custom path (${PMDK_INCLUDE_HINTS}).")
endif ()
if (NOT PMDK_LIBRARIES OR "${PMDK_LIBRARIES}" STREQUAL "")
    message(FATAL_ERROR "libmem not found in default locations or custom path (${PMDK_LIB_HINTS}).")
endif ()

message(STATUS "Found PMDK. Including ${PMDK_INCLUDE_DIRS} and linking ${PMDK_LIBRARIES}")
include_directories(${PMDK_INCLUDE_DIRS})


# libpmemobj + libpmempool
find_library(PMEMOBJ_LIBRARIES NAMES pmemobj libpmemobj HINTS ${PMDK_LIB_HINTS})
find_library(PMEMPOOL_LIBRARIES NAMES pmempool libpmempool HINTS ${PMDK_LIB_HINTS})

# libpmemobj++
# Use -DLIBPMEMOBJ++_PATH=/path/to/libpmemobj++ for custom path
set(LIBPMEMOBJ++_PATH "/scratch/pmem/libpmemobj-cpp" CACHE PATH
    "Path to custom libpmemobj++ install directory")
include_directories(${LIBPMEMOBJ++_PATH}/include)

# TBB
set(TBB_TEST OFF CACHE INTERNAL "")
set(TBB_EXAMPLES OFF CACHE INTERNAL "")
FetchContent_Declare(
    oneTBB
    GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
    GIT_TAG        v2021.1.1
    )
FetchContent_MakeAvailable(oneTBB)
include_directories(${oneTBB_SOURCE_DIR}/include)

# Dash
FetchContent_Declare(
    epoch_reclaimer
    GIT_REPOSITORY https://github.com/XiangpengHao/epoch-reclaimer.git
    GIT_TAG        master
    )
if (NOT epoch_reclaimer_POPULATED)
    FetchContent_Populate(epoch_reclaimer)
endif ()
include_directories(${epoch_reclaimer_SOURCE_DIR})
add_definitions(-DPMEM)
FetchContent_Declare(
    dash
    GIT_REPOSITORY https://github.com/baotonglu/dash.git
    GIT_TAG        7e1551f
    )
if (NOT dash_POPULATED)
    FetchContent_Populate(dash)
endif ()
include_directories(${dash_SOURCE_DIR}/src)
execute_process(COMMAND sed -i -e "s/, (void\\*)pool_addr//g" ${dash_SOURCE_DIR}/src/allocator.h)
execute_process(COMMAND sed -i -e "s/pmemobj_create_addr/pmemobj_create/g" ${dash_SOURCE_DIR}/src/allocator.h)
execute_process(COMMAND sed -i -e "s/pmemobj_open_addr/pmemobj_open/g" ${dash_SOURCE_DIR}/src/allocator.h)
execute_process(COMMAND sed -i -e "s#std::cout << \"Directory_Doubling#//#g" ${dash_SOURCE_DIR}/src/ex_finger.h)


set(OLD_CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH})
set(CMAKE_PREFIX_PATH "${LIBPMEMOBJ++_PATH};${VIPER_PMDK_PATH}")
set(PMEM_LIBS ${PMDK_LIBRARIES} ${PMEMOBJ_LIBRARIES} ${PMEMPOOL_LIBRARIES})

if (CMAKE_BUILD_TYPE MATCHES Release)
    message(STATUS "BUILDING BENCHMARKS IN RELEASE")
    add_compile_options(-O3 -Ofast -mtune=native)
else()
    message(WARNING "Building benchmarks in ${CMAKE_BUILD_TYPE}. This may impact performance!")
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
#find_package(OpenMP REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(put_key_value ../put_key_value.cpp ${BASE_BENCHMARK_FILES})
target_link_libraries(put_key_value viper ${PMEM_LIBS})
target_link_libraries(put_key_value uuid benchmark pmemkv aio pthread)
target_compile_features(put_key_value PUBLIC cxx_std_17)
set_target_properties(put_key_value PROPERTIES LINKER_LANGUAGE CXX)

add_executable(prefill_put ../prefill_put.cpp ${BASE_BENCHMARK_FILES})
target_link_libraries(prefill_put viper ${PMEM_LIBS})
target_link_libraries(prefill_put uuid benchmark pmemkv aio pthread)
target_compile_features(prefill_put PUBLIC cxx_std_17)
set_target_properties(prefill_put PROPERTIES LINKER_LANGUAGE CXX)

add_executable(same_key_update ../same_key_update.cpp ${BASE_BENCHMARK_FILES})
target_link_libraries(same_key_update viper ${PMEM_LIBS})
target_link_libraries(same_key_update uuid benchmark pmemkv aio pthread)
target_compile_features(same_key_update PUBLIC cxx_std_17)
set_target_properties(same_key_update PROPERTIES LINKER_LANGUAGE CXX)


add_executable(test_same_key3 ../test_same_key3.cpp ${BASE_BENCHMARK_FILES})
target_link_libraries(test_same_key3 viper ${PMEM_LIBS})
target_link_libraries(test_same_key3 uuid benchmark pmemkv aio pthread)
target_compile_features(test_same_key3 PUBLIC cxx_std_17)
set_target_properties(test_same_key3 PROPERTIES LINKER_LANGUAGE CXX)

add_executable(prefill_delete ../prefill_delete.cpp ${BASE_BENCHMARK_FILES})
target_link_libraries(prefill_delete viper ${PMEM_LIBS})
target_link_libraries(prefill_delete uuid benchmark pmemkv aio pthread)
target_compile_features(prefill_delete PUBLIC cxx_std_17)
set_target_properties(prefill_delete PROPERTIES LINKER_LANGUAGE CXX)

add_executable(put_latency ../put_latency.cpp ${BASE_BENCHMARK_FILES})
target_link_libraries(put_latency viper ${PMEM_LIBS})
target_link_libraries(put_latency uuid benchmark pmemkv aio pthread)
target_compile_features(put_latency PUBLIC cxx_std_17)
set_target_properties(put_latency PROPERTIES LINKER_LANGUAGE CXX)

add_executable(recovery_test ../recovery_test.cpp ${BASE_BENCHMARK_FILES})
target_link_libraries(recovery_test viper ${PMEM_LIBS})
target_link_libraries(recovery_test uuid benchmark pmemkv aio pthread)
target_compile_features(recovery_test PUBLIC cxx_std_17)
set_target_properties(recovery_test PROPERTIES LINKER_LANGUAGE CXX)

add_executable(put_rps ../put_rps.cpp ${BASE_BENCHMARK_FILES})
target_link_libraries(put_rps viper ${PMEM_LIBS})
target_link_libraries(put_rps uuid benchmark pmemkv aio pthread)
target_compile_features(put_rps PUBLIC cxx_std_17)
set_target_properties(put_rps PROPERTIES LINKER_LANGUAGE CXX)

add_executable(get_rps ../get_rps.cpp ${BASE_BENCHMARK_FILES})
target_link_libraries(get_rps viper ${PMEM_LIBS})
target_link_libraries(get_rps uuid benchmark pmemkv aio pthread)
target_compile_features(get_rps PUBLIC cxx_std_17)
set_target_properties(get_rps PROPERTIES LINKER_LANGUAGE CXX)

add_executable(ycsbA ../ycsbA.cpp ${BASE_BENCHMARK_FILES})
target_link_libraries(ycsbA viper ${PMEM_LIBS})
target_link_libraries(ycsbA uuid benchmark pmemkv aio pthread)
target_compile_features(ycsbA PUBLIC cxx_std_17)
set_target_properties(ycsbA PROPERTIES LINKER_LANGUAGE CXX)

add_executable(ycsbB ../ycsbB.cpp ${BASE_BENCHMARK_FILES})
target_link_libraries(ycsbB viper ${PMEM_LIBS})
target_link_libraries(ycsbB uuid benchmark pmemkv aio pthread)
target_compile_features(ycsbB PUBLIC cxx_std_17)
set_target_properties(ycsbB PROPERTIES LINKER_LANGUAGE CXX)

add_executable(get_key_value ../get_key_value.cpp ${BASE_BENCHMARK_FILES})
target_link_libraries(get_key_value viper ${PMEM_LIBS})
target_link_libraries(get_key_value uuid benchmark pmemkv aio pthread)
target_compile_features(get_key_value PUBLIC cxx_std_17)
set_target_properties(get_key_value PROPERTIES LINKER_LANGUAGE CXX)

add_executable(prefill_get ../prefill_get.cpp ${BASE_BENCHMARK_FILES})
target_link_libraries(prefill_get viper ${PMEM_LIBS})
target_link_libraries(prefill_get uuid benchmark pmemkv aio pthread)
target_compile_features(prefill_get PUBLIC cxx_std_17)
set_target_properties(prefill_get PROPERTIES LINKER_LANGUAGE CXX)

add_executable(prefill_update ../prefill_update.cpp ${BASE_BENCHMARK_FILES})
target_link_libraries(prefill_update viper ${PMEM_LIBS})
target_link_libraries(prefill_update uuid benchmark pmemkv aio pthread)
target_compile_features(prefill_update PUBLIC cxx_std_17)
set_target_properties(prefill_update PROPERTIES LINKER_LANGUAGE CXX)
