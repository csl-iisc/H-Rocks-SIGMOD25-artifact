#pragma once
#ifndef RECEIVER_H
#define RECEIVER_H
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include "command.h"

const int BATCH_SIZE = 20000000;

struct memtable {
    char* d_sortedKeys;
    char** d_sortedValuePointers;
    int* d_sortedOperationIDs;
    size_t size;

    memtable() : d_sortedKeys(nullptr), d_sortedValuePointers(nullptr), d_sortedOperationIDs(nullptr), size(0) {}

};

class Receiver {
    public:
        Receiver(); 
        Receiver(const std::queue<Command>& commands);
        Receiver(std::queue<Command>& readCommands, std::queue<Command>& writeCommands, std::queue<Command>& updateCommands);


        void receive(const Command& command);
        void operator()();
        void exit(); 

    private:
        std::queue<Command> readCommands;
        std::queue<Command> writeCommands;
        std::queue<Command> updateCommands;

        std::mutex queueMutex;
        std::condition_variable cv;
        bool shouldExit = false;

        void processBatch();
        void processQueue(const std::string& type, std::queue<Command>& commands);
        void processPuts(); 
        void processGets(); 
        void processRange(); 
        void processDeletes(); 
        void processUpdates(); 

        int keyLength = 0; 
        int valueLength = 0; 

        int readKeySize = 0;
        int writeKeySize = 0;
        int updateKeySize = 0;

        int readValueSize = 0;
        int writeValueSize = 0;
        int updateValueSize = 0;

        int numWrites = 0; 
        int numReads = 0; 
        int numUpdates = 0;

        char* putKeys = nullptr;
        char* putValues = nullptr;
        char* getKeys = nullptr;
        char* deleteKeys = nullptr;
        char* rangeKeys = nullptr;
        char* updateKeys = nullptr;

        int* putOperationIds = nullptr;
        int* getOperationIds = nullptr;
        int* deleteOperationIds = nullptr;
        int* rangeOperationIds = nullptr;
        int* updateOperationIds = nullptr;

        char** putValuePointers = nullptr;
        
        memtable immutableMemtable;
        memtable activeMemtable;


};
#endif // RECEIVER_H

