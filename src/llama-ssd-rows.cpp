#include "llama-ssd-rows.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <list>
#include <mutex>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

struct llama_ssd_rows::impl {
    static constexpr size_t page_size = 16384;
    struct page { uint64_t offset; std::vector<uint8_t> data; };
    int fd = -1;
    uint64_t offset, end, file_size;
    int64_t columns, rows;
    size_t row_bytes, capacity;
    ggml_to_float_t decode;
    mutable std::mutex mutex;
    std::list<page> lru;
    std::unordered_map<uint64_t, std::list<page>::iterator> pages;
    uint64_t reads = 0;

    ~impl() { if (fd >= 0) close(fd); }

    const page & get(uint64_t pos) {
        const uint64_t aligned = pos / page_size * page_size;
        auto found = pages.find(aligned);
        if (found != pages.end()) {
            lru.splice(lru.begin(), lru, found->second);
            return *lru.begin();
        }
        // Reuse the evicted allocation to keep cache memory bounded even during misses.
        if (lru.size() == capacity) {
            pages.erase(lru.back().offset);
            lru.splice(lru.begin(), lru, std::prev(lru.end()));
        } else {
            lru.push_front({0, std::vector<uint8_t>(page_size)});
        }
        auto & p = lru.front();
        p.offset = aligned;
        const size_t size = std::min<uint64_t>(page_size, file_size - aligned);
        size_t done = 0;
        while (done < size) {
            ssize_t n = pread(fd, p.data.data() + done, size - done, aligned + done);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) {
                lru.pop_front();
                throw std::runtime_error("PLE SSD read failed or file was truncated");
            }
            done += size_t(n);
            reads += uint64_t(n);
        }
        pages.emplace(aligned, lru.begin());
        return p;
    }
};

llama_ssd_rows::llama_ssd_rows(int fd, uint64_t offset, ggml_type type, int64_t columns,
                             int64_t rows, size_t cache_bytes) : pimpl(new impl) {
    auto & p = *pimpl;
    if (columns <= 0 || rows <= 0 || cache_bytes < impl::page_size ||
        type < 0 || type >= GGML_TYPE_COUNT || columns % ggml_blck_size(type)) {
        throw std::invalid_argument("invalid PLE SSD row geometry or cache budget");
    }
    p.row_bytes = ggml_row_size(type, columns);
    if (uint64_t(rows) > (std::numeric_limits<uint64_t>::max() - offset) / p.row_bytes) {
        throw std::invalid_argument("PLE SSD tensor extent overflows");
    }
    struct stat st{};
    if (fstat(fd, &st) != 0 || st.st_size < 0 || offset + uint64_t(rows)*p.row_bytes > uint64_t(st.st_size)) {
        throw std::invalid_argument("PLE SSD tensor exceeds source file");
    }
    p.decode = ggml_get_type_traits(type)->to_float;
    if (!p.decode && type != GGML_TYPE_F32) throw std::invalid_argument("unsupported PLE SSD type");
    p.offset = offset; p.end = offset + uint64_t(rows)*p.row_bytes; p.file_size = st.st_size;
    p.columns = columns; p.rows = rows; p.capacity = cache_bytes / impl::page_size;
    p.fd = dup(fd);
    if (p.fd < 0) throw std::runtime_error("cannot duplicate PLE SSD descriptor");
#ifdef __APPLE__
    if (fcntl(p.fd, F_NOCACHE, 1) != 0 || fcntl(p.fd, F_RDAHEAD, 0) != 0) {
        throw std::runtime_error("cannot disable filesystem caching for PLE SSD reads");
    }
#endif
}

llama_ssd_rows::~llama_ssd_rows() = default;

void llama_ssd_rows::read_rows(const int32_t * ids, size_t count, float * output) {
    auto & p = *pimpl;
    if (count && (!ids || !output)) throw std::invalid_argument("null PLE SSD input/output");
    for (size_t i = 0; i < count; ++i) {
        if (ids[i] < 0 || ids[i] >= p.rows) throw std::out_of_range("PLE SSD row index");
    }
    std::lock_guard<std::mutex> lock(p.mutex);
    std::vector<uint8_t> row(p.row_bytes);
    for (size_t i = 0; i < count; ++i) {
        const uint64_t pos = p.offset + uint64_t(ids[i])*p.row_bytes;
        size_t done = 0;
        while (done < p.row_bytes) {
            const auto & page = p.get(pos + done);
            const size_t within = pos + done - page.offset;
            const size_t n = std::min(p.row_bytes - done, impl::page_size - within);
            memcpy(row.data() + done, page.data.data() + within, n);
            done += n;
        }
        if (p.decode) p.decode(row.data(), output + i*p.columns, p.columns);
        else memcpy(output + i*p.columns, row.data(), p.row_bytes);
    }
}
size_t llama_ssd_rows::cached_bytes() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->lru.size()*impl::page_size;
}
uint64_t llama_ssd_rows::bytes_read() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->reads;
}
