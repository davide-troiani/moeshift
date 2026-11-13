#include "llama-kv-cache.h"
#include "llama-model.h"
#include "llama-io.h"
#include "ggml-backend.h"

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

static void check(bool ok) {
    if (!ok) throw std::runtime_error("KV state compatibility check failed");
}

struct state_io : llama_io_read_i, llama_io_write_i {
    std::vector<uint8_t> bytes;
    size_t offset = 0;
    void write(const void * src, size_t size) override {
        const auto * data = static_cast<const uint8_t *>(src);
        bytes.insert(bytes.end(), data, data + size);
    }
    void read(void * dst, size_t size) override {
        check(size <= bytes.size() - offset);
        std::memcpy(dst, bytes.data() + offset, size);
        offset += size;
    }
    void write_tensor(ggml_tensor * t, size_t off, size_t size) override {
        std::vector<uint8_t> data(size);
        ggml_backend_tensor_get(t, data.data(), off, size);
        write(data.data(), size);
    }
    void read_tensor(ggml_tensor * t, size_t off, size_t size) override {
        std::vector<uint8_t> data(size);
        read(data.data(), size);
        ggml_backend_tensor_set(t, data.data(), off, size);
    }
    size_t n_bytes() override { return offset; }
};

int main(int argc, char ** argv) {
    check(argc == 2);
    unsetenv("LLAMA_KV_GROW");
    state_io baseline;
    std::ifstream fixture(argv[1]);
    check(fixture.good());
    unsigned byte;
    while (fixture >> std::hex >> byte) {
        check(byte <= 255);
        baseline.bytes.push_back(uint8_t(byte));
    }
    check(baseline.bytes.size() == 316);
    for (auto arch : {LLM_ARCH_QWEN2VL, LLM_ARCH_QWEN3VL, LLM_ARCH_QWEN35MOE, LLM_ARCH_QWEN4EXP}) {
        std::unique_ptr<llama_model> model(llama_model_create(arch, llama_model_default_params()));
        auto & hp = model->hparams;
        hp = llama_hparams{};
        hp.n_layer = 1;
        hp.n_embd = 32;
        hp.n_head_arr.fill(1);
        hp.n_head_kv_arr.fill(1);
        hp.n_embd_head_k_full = hp.n_embd_head_v_full = 32;
        hp.rope_type = LLAMA_ROPE_TYPE_IMROPE;
        state_io expected = baseline;
        if (arch == LLM_ARCH_QWEN4EXP) {
            // Preserve the existing Qwen4Exp token,x,y record, including PLE token history.
            hp.ple_n_heads = 1;
            const llama_token tok = 42;
            const auto * data = reinterpret_cast<const uint8_t *>(&tok);
            expected.bytes.insert(expected.bytes.begin() + 16, data, data + sizeof(tok));
        }
        for (llama_seq_id destination : {-1, 1}) {
            llama_kv_cache cache(*model, hp, GGML_TYPE_F32, GGML_TYPE_F32, false, false, true,
                    32, 2, 1, 0, LLAMA_SWA_TYPE_NONE, nullptr, nullptr);
            state_io input = expected;
            cache.state_read(input, destination, 0);
            check(input.offset == input.bytes.size());
            const auto & cells = cache.get_cells(destination == -1 ? 0 : destination);
            check(cells.pos_get(0) == 5);
            check(cells.ext_get(0).x == 7 && cells.ext_get(0).y == 11);
            if (arch == LLM_ARCH_QWEN4EXP) check(cells.ext_get(0).tok == 42);
            state_io output;
            cache.state_write(output, destination, 0);
            auto wanted = expected.bytes;
            if (destination != -1) {
                const size_t seq_offset = arch == LLM_ARCH_QWEN4EXP ? 28 : 24;
                std::memcpy(wanted.data() + seq_offset, &destination, sizeof(destination));
            }
            check(output.bytes == wanted);
        }
    }
}
