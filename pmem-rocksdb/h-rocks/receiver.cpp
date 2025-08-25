#include "receiver.h"
#include <iostream>
#include <cstring>
#include "gpu_puts.cuh"


// Initialize memtable with size 0
//memtable immutableMemtable {nullptr, nullptr, nullptr, 0};

Receiver::Receiver(std::queue<Command>& readCommands, std::queue<Command>& writeCommands, std::queue<Command>& updateCommands)
    : readCommands(readCommands), writeCommands(writeCommands), updateCommands(updateCommands) {
    }

void Receiver::receive(const Command& command) {
    std::cout << "Receiving\n"; 
    std::unique_lock<std::mutex> lock(queueMutex);
    if (command.type == Type::GET) {
        readCommands.push(command);
        readKeySize += command.key.size();

    } else if (command.type == Type::PUT) {
        writeCommands.push(command);
        writeKeySize += command.key.size();
        writeValueSize += command.value.size();

    } else if (command.type == Type::DELETE) {
        writeCommands.push(command);
        writeKeySize += command.key.size();

    } else if (command.type == Type::RANGE) {
        readCommands.push(command);
        readKeySize += command.key.size();

    } else if (command.type == Type::UPDATE) {
        updateCommands.push(command);
        updateKeySize += command.key.size();
        updateValueSize += command.value.size();

    }
    if (writeCommands.size() + readCommands.size() + updateCommands.size() >= BATCH_SIZE) {
        cv.notify_one();

    }

}
void Receiver::operator()() {
    while (true) {
        std::unique_lock<std::mutex> lock(queueMutex);
        cv.wait(lock, 
                [this] {
                return (readCommands.size() + writeCommands.size() + updateCommands.size() >= BATCH_SIZE) || shouldExit; });
        // Exit condition: When the shouldExit flag is set
        if (shouldExit) {
            break;
        }
        processBatch();
    }
}

void Receiver::exit() {
    std::unique_lock<std::mutex> lock(queueMutex);
    shouldExit = true;
    cv.notify_one();
}

void Receiver::processBatch() {
    std::cout << "Here\n"; 
    char* putKeys = new char[writeKeySize];
    char* putValues = new char[writeValueSize];
    char* getKeys = new char[readKeySize];
    char* deleteKeys = new char[writeKeySize];
    char* rangeKeys = new char[readKeySize];
    char* updateKeys = new char[updateKeySize];
    char* updateValues = new char[updateValueSize];

    int batchSize = 0;
    int putKeyIndex = 0, getKeyIndex = 0, deleteKeyIndex = 0, rangeKeyIndex = 0, updateKeyIndex = 0, putValueIndex = 0;
    while (batchSize < BATCH_SIZE) {

        while (!readCommands.empty() || !writeCommands.empty() || !updateCommands.empty()) {
            if (!readCommands.empty()) {
                Command cmd = readCommands.front();
                readCommands.pop();
                if (cmd.type == Type::GET) {
                    strcpy(getKeys + getKeyIndex, cmd.key.c_str());
                    getOperationIds[batchSize] = cmd.operationID;
                    getKeyIndex += cmd.key.length() + 1;

                } else if (cmd.type == Type::RANGE) {
                    strcpy(rangeKeys + rangeKeyIndex, cmd.key.c_str());
                    rangeOperationIds[batchSize] = cmd.operationID;
                    rangeKeyIndex += cmd.key.length() + 1;
                }
                batchSize++;
                numReads++; 
            }

            if (!writeCommands.empty()) {
                Command cmd = writeCommands.front();
                writeCommands.pop();
                if (cmd.type == Type::PUT) {
                    strcpy(putKeys + putKeyIndex, cmd.key.c_str());
                    strcpy(putValues + putValueIndex, cmd.value.c_str());
                    putKeyIndex += cmd.key.length() + 1;
                    putValueIndex += cmd.key.length() + 1;
                    keyLength = cmd.key.length() + 1; 
                    valueLength = cmd.value.length() + 1; 
                    putValuePointers[batchSize] = strdup(cmd.value.c_str());
                    putOperationIds[batchSize] = cmd.operationID;

                } else if (cmd.type == Type::DELETE) {
                    strcpy(deleteKeys + deleteKeyIndex, cmd.key.c_str());
                    deleteKeyIndex += cmd.key.length() + 1;
                    deleteOperationIds[batchSize] = cmd.operationID;
                }
                batchSize++;
                numWrites++; 
            }
            if (!updateCommands.empty()) {
                Command cmd = updateCommands.front();
                updateCommands.pop();
                strcpy(updateKeys + updateKeyIndex, cmd.key.c_str());
                updateOperationIds[batchSize] = cmd.operationID;
                batchSize++;
                numUpdates++; 

            }
        }
    }

    // Start processing 

    processPuts(); 
    processDeletes(); 
    processGets(); 
    processRange(); 
    processUpdates(); 

    writeKeySize = 0;
    writeValueSize = 0;
    readKeySize = 0;
    updateKeySize = 0;
    updateValueSize = 0;

    delete[] putKeys;
    delete[] putValues;
    delete[] getKeys;
    delete[] deleteKeys;
    delete[] rangeKeys;
    delete[] updateKeys;
    delete[] updateValues;
}

void Receiver::processPuts()
{
    std::cout << "Here\n"; 
    sortPutsOnGPU(putKeys, putValuePointers, putOperationIds, numWrites, keyLength, activeMemtable);  
}
void Receiver::processGets()
{
}
void Receiver::processDeletes()
{
}
void Receiver::processRange()
{
}
void Receiver::processUpdates()
{
}

