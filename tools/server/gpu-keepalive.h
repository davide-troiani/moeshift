#pragma once
#include <atomic>
#include <thread>
class gpu_keepalive {
public:
    gpu_keepalive();
    ~gpu_keepalive();
    bool ready = false;
private:
    std::atomic<bool> stopped{false};
    std::thread worker;
};
