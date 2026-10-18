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
#ifdef __cplusplus
}
#endif
