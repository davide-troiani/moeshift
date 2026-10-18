#include "bridge.h"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <unordered_map>
#include <cassert>
struct ticket {
    uint64_t value = 0;
    uint64_t done = 0;
    std::mutex mutex;
    std::condition_variable cv;
};
static std::mutex registry_mutex;
static std::unordered_map<void *, ticket *> registry;
void * les_io_create(void) { return new ticket; }
void les_io_destroy(void * handle) {
    auto * t = static_cast<ticket *>(handle);
    if (!t) return;
    std::lock_guard<std::mutex> lock(registry_mutex);
    for (auto i = registry.begin(); i != registry.end();) {
        if (i->second == t) i = registry.erase(i); else ++i;
    }
    delete t;
}
void les_io_register(void * handle, void * data) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto result = registry.emplace(data, static_cast<ticket *>(handle));
    assert(result.second || result.first->second == handle);
}
uint64_t les_io_begin(void * handle) {
    auto * t = static_cast<ticket *>(handle);
    std::lock_guard<std::mutex> lock(t->mutex);
    return ++t->value;
}
void les_io_signal(void * handle, uint64_t value) {
    auto * t = static_cast<ticket *>(handle);
    { std::lock_guard<std::mutex> lock(t->mutex);t->done=value; }
    t->cv.notify_all();
}
void les_io_wait_for_tensor(void * data) {
    ticket * t;
    {
        std::lock_guard<std::mutex> lock(registry_mutex);
        auto it=registry.find(data);
        if (it==registry.end()) return;
        t=it->second;
    }
    std::unique_lock<std::mutex> lock(t->mutex);
    t->cv.wait(lock,[&]{return t->done>=t->value;});
}
