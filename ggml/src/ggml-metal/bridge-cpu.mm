#include "bridge.h"
#import <Metal/Metal.h>
#include <mutex>
#include <unordered_map>
#include <cassert>
struct ticket {
    id<MTLSharedEvent> event = nil;
    uint64_t value = 0;
    std::mutex mutex;
};
static std::mutex registry_mutex;
static std::unordered_map<void *, ticket *> registry;

void * les_io_create(void) {
    static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    auto * t = new ticket;
    t->event = [device newSharedEvent];
    if (!t->event) abort();
    return t;
}
void les_io_destroy(void * handle) {
    auto * t = static_cast<ticket *>(handle);
    if (!t) return;
    std::lock_guard<std::mutex> lock(registry_mutex);
    for (auto i = registry.begin(); i != registry.end();) {
        if (i->second == t) i = registry.erase(i); else ++i;
    }
    [t->event release];
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
    t->event.signaledValue = value;
}
void * les_io_lookup(void * data, uint64_t * value) {
    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    const auto it = registry.find(data);
    if (it == registry.end()) return nullptr;
    ticket * t = it->second;
    std::lock_guard<std::mutex> lock(t->mutex);
    *value = t->value;
    return t;
}
void les_io_encode_wait(void * handle, uint64_t value, void * command_buffer) {
    ticket * t = static_cast<ticket *>(handle);
    id<MTLCommandBuffer> cb = (id<MTLCommandBuffer>) command_buffer;
    [cb encodeWaitForEvent:t->event value:value];
}
