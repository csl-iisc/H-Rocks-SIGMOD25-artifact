#include "memtable.cuh"

void deleteMemtable(struct Memtable *mt) 
{
    cudaFree(mt->d_sortedKeys); 
    cudaFree(mt->d_sortedValuePointers); 
    cudaFree(mt->d_sortedOperationIDs); 
}
