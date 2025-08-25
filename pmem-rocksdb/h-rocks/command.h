#ifndef COMMAND_H
#define COMMAND_H

#include <string>
#include <iostream>
#include <vector>

enum Type { GET, PUT, DELETE, RANGE, UPDATE  };
class Command {
    public:
        Type type;
        std::string key;
        std::string startKey;
        std::string endKey;
        std::string value;
        int numElems;
        uint64_t operationID; 

        Command(Type type, const std::string& key, uint64_t operationID)
            : type(type), key(key), value(""), numElems(0), operationID(operationID) {}
        Command(Type type, const std::string& key, const std::string& value, uint64_t operationID)
            : type(type), key(key), value(value), numElems(0), operationID(operationID) {}
        Command(Type type, const std::string& startKey, const std::string& endKey, int numElems,  uint64_t operationID)
            : type(type), startKey(startKey), endKey(endKey), numElems(0), operationID(operationID) {}


};

class PutCommand {
    public: 
    std::vector<char> tKeys; 
    std::vector<char> tValues; 
    std::vector<uint64_t> tOpID;
    char* keys;
    char* values; 
    char** valuePtrs; 
    uint64_t* operationIDs; 
    int numElems;
    int keyLength;
    int valueLength; 
    uint64_t numPuts; 
    PutCommand() {
        numPuts = 0; 
    }
}; 

class DeleteCommand {
    public: 
    std::vector<char> tKeys; 
    std::vector<char> tValues; 
    std::vector<uint64_t> tOpID;
    char* keys;
    char* values; 
    char** valuePtrs; 
    uint64_t* operationIDs; 
    int numElems;
    int keyLength;
    int valueLength; 
    uint64_t numDeletes; 
    DeleteCommand() {
        numDeletes = 0; 
    }
}; 

class GetCommand {
    public: 
    std::vector<char> tKeys; 
    std::vector<uint64_t> tOpID;
    char* keys;
    char** valuePtrs; 
    uint64_t* operationIDs; 
    int numElems;
    int keyLength;
    uint64_t numGets; 
    bool hasUpdates; 
    GetCommand() {
        hasUpdates = false; 
        numGets = 0; 
    }
}; 

class MergeCommand{
    public: 
    std::vector<char> tKeys; 
    std::vector<uint64_t> tOpID;
    char* keys;
    char* values; 
    int* locks;
    uint64_t* operationIDs; 
    int numElems;
    int keyLength;
    int valueLength; 
    uint64_t numMerges; 
    MergeCommand() {
        numMerges = 0; 
    }
}; 

class RangeCommand {
    public: 
    std::vector<char> tStartKeys; 
    std::vector<char> tEndKeys; 
    std::vector<uint64_t> tOpID;
    char* startKeys;
    char* endKeys;
    char** valuePtrs; 
    uint64_t* operationIDs; 
    int numElems;
    int keyLength;
    int valueLength; 
    uint64_t numRangeQs; 
    bool hasUpdates; 
    RangeCommand() {
        numRangeQs = 0; 
        hasUpdates = false;
    }
}; 

#endif  // COMMAND_H



/*
#pragma once

#include <string>

struct Command {
enum Type { PUT, GET, DELETE, UPDATE, RANGE };
Type type;
uint64_t operationID;
std::string key;
std::string value;
int numElems; 
};
*/
