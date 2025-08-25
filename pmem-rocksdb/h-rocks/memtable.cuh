#pragma once
#include <iostream>

struct Memtable
{
    char* d_sortedKeys;
    char** d_sortedValuePointers;
    uint64_t* d_sortedOperationIDs;
    uint64_t size; // Size of the memtable
    int batchID; // memtableID 
    unsigned int* locks; 
    unsigned int keyLength; 
    unsigned int valueLength; 
    Memtable() {
        size = 0;
        batchID = -1; 
    }
}; 

struct UpdateMemtable 
{
    char* d_sortedKeys;
    char* d_sortedValues; 
    uint64_t* d_sortedOperationIDs;
    uint64_t size; // Size of the memtable
    int batchId; // memtableID 
    unsigned int keyLength; 
    unsigned int valueLength; 
    UpdateMemtable() {
        size = 0; 
        batchId = -1;
        keyLength = 8; 
        valueLength = 8;
    }
}; 

struct MemtableWithValues
{
    char* d_sortedKeys;
    char* d_sortedValues;
    char** d_sortedValuePointers;
    uint64_t* d_sortedOperationIDs;
    uint64_t size; // Size of the memtable
    int batchID; // memtableID 
    unsigned int* locks; 
    unsigned int keyLength; 
    unsigned int valueLength; 
    MemtableWithValues() {
        size = 0;
        batchID = -1; 
    }
}; 


void deleteMemtable(struct Memtable *mt); 
