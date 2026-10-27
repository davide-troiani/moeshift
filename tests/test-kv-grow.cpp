#include "llama-kv-cache.h"
#include "llama-model.h"
#include "ggml-backend.h"
#include <cstdlib>
#include <stdexcept>
#include <vector>

static void check(bool ok) {
    if (!ok) throw std::runtime_error("KV growth self-check failed");
}

int main() {
    setenv("LLAMA_KV_GROW", "1", 1);
    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_LLAMA, llama_model_default_params()));
    auto & hp = model->hparams;
    hp = llama_hparams{};
    hp.n_layer = 1;
    hp.n_embd = 32;
    hp.n_head_arr.fill(1);
    hp.n_head_kv_arr.fill(1);
    hp.n_embd_head_k_full = hp.n_embd_head_v_full = 32;
    for (auto type : { GGML_TYPE_F32, GGML_TYPE_Q8_0 }) {
        llama_kv_cache cache(*model, hp, type, type, false, false, true,
                8192, 1, 1, 0, LLAMA_SWA_TYPE_NONE, nullptr, nullptr);
        check(cache.get_size() == 2048);
        llama_kv_cache::slot_info slot;
        slot.s0 = slot.s1 = 0;
        slot.resize(1);
        slot.strm = {0};
        slot.idxs[0] = {0};
        ggml_init_params params = { 32*ggml_tensor_overhead(), nullptr, true };
        ggml_context_ptr ctx(ggml_init(params));
        auto * k = cache.get_k(ctx.get(), 0, 2048, slot);
        auto * v = cache.get_v(ctx.get(), 0, 2048, slot);
        std::vector<uint8_t> original(ggml_nbytes(k), 0x3f);
        ggml_backend_tensor_set(k, original.data(), 0, original.size());
        ggml_backend_tensor_set(v, original.data(), 0, original.size());
        for (uint32_t needed : {2049u, 4097u}) {
            check(cache.ensure_capacity(needed));
            check(cache.get_size() == (needed == 2049 ? 4096 : 8192));
            for (auto * tensor : {cache.get_k(ctx.get(), 0, cache.get_size(), slot),
                                 cache.get_v(ctx.get(), 0, cache.get_size(), slot)}) {
                std::vector<uint8_t> actual(ggml_nbytes(tensor));
                ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size());
                for (size_t i = 0; i < actual.size(); ++i) {
                    check(actual[i] == (i < original.size() ? 0x3f : 0));
                }
            }
        }
        check(!cache.ensure_capacity(8193));
        check(cache.get_size() == 8192);
        cache.clear(true);
        auto * zero = cache.get_k(ctx.get(), 0, 8192, slot);
        std::vector<uint8_t> cleared(ggml_nbytes(zero), 1);
        ggml_backend_tensor_get(zero, cleared.data(), 0, cleared.size());
        for (uint8_t b : cleared) check(b == 0);
    }
    llama_kv_cache eager(*model, hp, GGML_TYPE_F32, GGML_TYPE_F32, true, false, true,
            8192, 1, 1, 0, LLAMA_SWA_TYPE_NONE, nullptr, nullptr);
    check(eager.get_size() == 8192);
}
