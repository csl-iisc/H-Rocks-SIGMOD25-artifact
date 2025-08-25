#!/bin/bash

g++ -O3 -g -std=c++11 -m64 -D_REENTRANT -fno-strict-aliasing -I./atomic_ops -DINTEL -Wno-unused-value -Wno-format  -o ./main-gu-zipfian main-gu-zipfian.c -m64 -lpmemobj -lpmem -lpthread -DUSE_PMDK

g++ -O3 -std=c++11 -m64 -D_REENTRANT -fno-strict-aliasing -I./atomic_ops -DINTEL -Wno-unused-value -Wno-format  -o ./insert_test insert_test.cc -m64 -lpmemobj -lpmem -lpthread -fopenmp -DUSE_PMDK

g++ -O3 -std=c++11 -m64 -D_REENTRANT -fno-strict-aliasing -I./atomic_ops -DINTEL -Wno-unused-value -Wno-format  -o ./get_test get_test.cc -m64 -lpmemobj -lpmem -lpthread -fopenmp -DUSE_PMDK

g++ -O3 -std=c++11 -m64 -D_REENTRANT -fno-strict-aliasing -I./atomic_ops -DINTEL -Wno-unused-value -Wno-format  -o ./remove_test remove_test.cc -m64 -lpmemobj -lpmem -lpthread -fopenmp -DUSE_PMDK

g++ -O3 -std=c++11 -m64 -D_REENTRANT -fno-strict-aliasing -I./atomic_ops -DINTEL -Wno-unused-value -Wno-format  -o ./update_test update_test.cc -m64 -lpmemobj -lpmem -lpthread -fopenmp -DUSE_PMDK

g++ -O3 -g -std=c++11 -m64 -D_REENTRANT -fno-strict-aliasing -I./atomic_ops -DINTEL -Wno-unused-value -Wno-format  -o ./concurrency_test concurrency_test.cc -m64 -lpmemobj -lpmem -lpthread -fopenmp -DUSE_PMDK

g++ -g -std=c++11 -m64 -D_REENTRANT -fno-strict-aliasing -I./atomic_ops -DINTEL -Wno-unused-value -Wno-format  -o ./concurrency_test2 concurrency_test2.cc -m64 -lpmemobj -lpmem -lpthread -fopenmp -DUSE_PMDK

g++ -g -std=c++11 -m64 -D_REENTRANT -fno-strict-aliasing -I./atomic_ops -DINTEL -Wno-unused-value -Wno-format  -o ./lost_update lost_update.cc -m64 -lpmemobj -lpmem -lpthread -fopenmp -DUSE_PMDK
