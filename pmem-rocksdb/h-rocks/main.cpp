#include <iostream>
#include <string>
#include <thread>
#include "sender.h"
#include "receiver.h"
#include "command.h"

int main() {
    std::queue<Command> readCommands, writeCommands, updateCommands;
    Receiver receiver(readCommands, writeCommands, updateCommands);
    Sender sender(readCommands, writeCommands, updateCommands);

    //std::thread senderThread(std::ref(sender));
    std::thread receiverThread(std::ref(receiver));
    std::this_thread::sleep_for(std::chrono::seconds(1));

    for (int i = 0; i < BATCH_SIZE * 3; i++) {
        sender.Put("key" + std::to_string(i), "value" + std::to_string(i));
        //std::cout << "i: " << i << "\t"; 
    }
    receiver.exit();
    receiverThread.join();

    return 0;
}
