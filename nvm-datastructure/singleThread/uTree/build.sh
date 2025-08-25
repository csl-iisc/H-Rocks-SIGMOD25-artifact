#!/bin/bash

g++ -O3 -march=native -std=c++11 -DNDEBUG -m64 -D_REENTRANT -fno-strict-aliasing -DINTEL -Wno-unused-value -Wno-format  -o ./main  ./main.cpp -lprofiler -ggdb

g++ -O3 -march=native -std=c++11 -DNDEBUG -m64 -D_REENTRANT -fno-strict-aliasing -DINTEL -Wno-unused-value -Wno-format  -o ./concurrency_test ./concurrency_test.cc -lprofiler -ggdb -DUSE_PMDK
