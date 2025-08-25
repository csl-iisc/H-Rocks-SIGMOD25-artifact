#pragma once

#include <string>
#include "receiver.h"
#include "command.h"

class Sender {
    public:
        //Sender(Receiver& receiver);
        Sender(std::queue<Command>& readCommands, std::queue<Command>& writeCommands, std::queue<Command>& updateCommands);
        void Get(const std::string& key);
        void Put(const std::string& key, const std::string& value);
        void Delete(const std::string& key);
        void Range(const std::string& key, int numElems);
        void Update(const std::string& key);
        void exit();

    private:
        Receiver receiver(std::queue<Command>& readCommands, std::queue<Command>& writeCommands, std::queue<Command>& updateCommands);
        void send(Type type, const std::string& key, const std::string& value = "", int num_elems = 0);
        std::queue<Command>& readCommands;
        std::queue<Command>& writeCommands;
        std::queue<Command>& updateCommands;
        std::mutex queueMutex;
        std::condition_variable cv;       
        unsigned int operationID;
};

