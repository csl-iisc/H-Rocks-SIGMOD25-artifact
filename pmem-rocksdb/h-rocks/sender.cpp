#include "sender.h"
#include "receiver.h"
#include <iostream>

Sender::Sender(std::queue<Command>& readCommands, std::queue<Command>& writeCommands, std::queue<Command>& updateCommands)
    : readCommands(readCommands), writeCommands(writeCommands), updateCommands(updateCommands), operationID(0) {}

void Sender::Put(const std::string& key, const std::string& value) {
    Command command(Type::PUT, key, value, operationID++);
    std::cout << "Putting.. " << operationID << "\n"; 
    writeCommands.push(command);
}

void Sender::Get(const std::string& key) {
    Command command(Type::GET, key, operationID++);
    readCommands.push(command);
}

void Sender::Delete(const std::string& key) {
    Command command(Type::DELETE, key, operationID++);
    writeCommands.push(command);
}

void Sender::Range(const std::string& key, int num_elems) {
    Command command(Type::RANGE, key, num_elems, operationID++);
    readCommands.push(command);
}

void Sender::Update(const std::string& key) {
    Command command(Type::UPDATE, key, operationID++);
    updateCommands.push(command);
}

void Sender::exit() {
    //receiver.exit();
}
/*
   void Sender::send(Command::Type type, const std::string& key, const std::string& value, int num_elems) {
   std::unique_lock<std::mutex> lock(queueMutex);
   Command command{type, key, value, num_elems, operationID++};

   if (type == Command::GET || type == Command::RANGE) {
   readCommands.push(command);

   } else if (type == Command::PUT || type == Command::DELETE) {
   writeCommands.push(command);

   } else if (type == Command::UPDATE) {
   updateCommands.push(command);

   }

   cv.notify_one();

   }

   void Sender::Get(const std::string& key) {
   Command cmd;
   cmd.type = "GET";
   cmd.key = key;
   cmd.operationID = operationID++;
   readCommands.push(cmd);

   }

   void Sender::Put(const std::string& key, const std::string& value) {
   Command cmd;
   cmd.type = "PUT";
   cmd.key = key;
   cmd.value = value;
   cmd.operationID = operationID++;
   writeCommands.push(cmd);

   }

   void Sender::Delete(const std::string& key) {
   Command cmd;
   cmd.type = "DELETE";
   cmd.key = key;
   cmd.operationID = operationID++;
   writeCommands.push(cmd);

   }

   void Sender::Range(const std::string& key, int num_elems) {
   Command cmd;
   cmd.type = "RANGE";
   cmd.key = key;
   cmd.numElems = num_elems;
   cmd.operationID = operationID++;
   readCommands.push(cmd);

   }

   void Sender::Update(const std::string& key) {
   Command cmd;
   cmd.type = "UPDATE";
   cmd.key = key;
   cmd.operationID = operationID++;
   updateCommands.push(cmd);

   }

   Sender::Sender(Receiver& receiver) : receiver(receiver), nextOperationId(1) {}

   void Sender::Put(const std::string& key, const std::string& value) {
   Command cmd = {Command::PUT, nextOperationId++, key, value};
receiver.receive(cmd);
}

void Sender::Get(const std::string& key) {
    Command cmd = {Command::GET, nextOperationId++, key, ""};
    receiver.receive(cmd);
}

void Sender::Delete(const std::string& key) {
    Command cmd = {Command::DELETE, nextOperationId++, key, ""};
    receiver.receive(cmd);
}

void Sender::Range(const std::string& key, int rangeElems) {
    Command cmd = {Command::RANGE, nextOperationId++, key, "", rangeElems};
    receiver.receive(cmd);
}

void Sender::Update(const std::string& key) {
    Command cmd = {Command::UPDATE, nextOperationId++, key, ""};
    receiver.receive(cmd);
}

*/

