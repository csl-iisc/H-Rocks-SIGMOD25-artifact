#include "gpu_puts.cuh"
#include "batch.h"
#include <thread>
#include <chrono>

int main() {
    std::queue<Command> readCommands, writeCommands, updateCommands;
    Batch batch(readCommands, writeCommands, updateCommands, 0);

    for (int i = 0; i < 10000; ++i) {
        std::string key = "sameKey"; 
        std::string value = "sameVal"; 
        batch.Put(key, value);
        batch.Get(key);
        //batch.Get("key" + std::to_string(i));
        /*
        if (i % 10 == 0) {
            batch.Delete("key" + std::to_string(i));
        }
        */
    }
    
    //std::this_thread::sleep_for(std::chrono::seconds(5));

    batch.Execute();

    return 0;
}
