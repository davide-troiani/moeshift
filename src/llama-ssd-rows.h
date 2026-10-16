#pragma once

#include "ggml.h"
#include <cstddef>
#include <cstdint>
#include <memory>

// Explicit row reads keep large PLE tables out of model and Metal allocations.
class llama_ssd_rows {
public:
    llama_ssd_rows(int fd, uint64_t offset, ggml_type type, int64_t columns,
                   int64_t rows, size_t cache_bytes);
    ~llama_ssd_rows();
    void read_rows(const int32_t * ids, size_t count, float * output);
    size_t cached_bytes() const;
    uint64_t bytes_read() const;
private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
