#include "ggml-backend.h"
#include "ggml-metal.h"
#include <cstdio>
#include <cstdlib>

static void check(bool ok) { if (!ok) std::abort(); }

int main() {
    ggml_backend_load_all();
    auto dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    check(dev != nullptr);
    auto buffer = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dev), 1024*1024);
    check(buffer != nullptr);
    void * address = ggml_backend_buffer_get_base(buffer);
    size_t offsets[] = {0, 512*1024}, sizes[] = {64*1024, 64*1024};
    check(ggml_backend_metal_buffer_active_ranges(buffer, offsets, sizes, 2));
    check(ggml_backend_buffer_get_base(buffer) == address);
    size_t invalid_offset = 1024*1024, invalid_size = 1;
    check(!ggml_backend_metal_buffer_active_ranges(buffer, &invalid_offset, &invalid_size, 1));
    check(!ggml_backend_metal_buffer_active_ranges(buffer, nullptr, nullptr, 2));
    check(ggml_backend_metal_buffer_active_ranges(buffer, nullptr, nullptr, 0));
    check(ggml_backend_buffer_get_base(buffer) == address);
    check(ggml_backend_metal_buffer_active_ranges(buffer, offsets, sizes, 2));
    ggml_backend_buffer_free(buffer);
    std::puts("PASS: shrink, restore, shrink; stable address; invalid ranges rejected");
}
