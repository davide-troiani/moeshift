#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
__attribute__((visibility("default"))) void * les_io_create(void);
__attribute__((visibility("default"))) void les_io_destroy(void * ticket);
__attribute__((visibility("default"))) void les_io_register(void * ticket, void * data);
__attribute__((visibility("default"))) uint64_t les_io_begin(void * ticket);
__attribute__((visibility("default"))) void les_io_signal(void * ticket, uint64_t value);
void * les_io_lookup(void * data, uint64_t * value);
void les_io_encode_wait(void * ticket, uint64_t value, void * command_buffer);
void les_io_wait_for_tensor(void * data);
enum les_io_event_type {
    LES_IO_WAIT_ENTER = 1,
    LES_IO_WAIT_EXIT = 2,
    LES_IO_WAVE_CREATE = 3,
    LES_IO_WAVE_JOIN = 4,
    LES_IO_READ_COMPLETE = 5,
    LES_IO_EXPERTS_AVAILABLE = 6,
};
struct les_io_event {
    uint64_t timestamp_ns;
    uint64_t step;
    uint64_t ticket;
    uint64_t bytes_demand;
    uint32_t layer;
    uint32_t misses;
    uint32_t reads_completed;
    uint32_t type;
};
// Copies up to capacity oldest buffered events; returns the number copied.
__attribute__((visibility("default"))) uint32_t les_io_drain_events(struct les_io_event * out, uint32_t capacity);
void les_io_record_event(uint32_t type, uint32_t layer, uint64_t step, uint64_t ticket,
                         uint32_t misses, uint64_t bytes_demand, uint32_t reads_completed);
void les_io_register_layer(void * handle, void * data, uint32_t layer);
void les_io_set_step(void * handle, uint64_t step);
#ifdef __cplusplus
}
#endif
