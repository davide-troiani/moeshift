#include "llama-memory-hybrid-idx.h"
#include "llama-model.h"
#include "ggml-backend.h"
#include <cstdlib>
#include <stdexcept>
#include <vector>

static void check(bool ok) {
    if (!ok) throw std::runtime_error("indexer key-only cache check failed");
}

static size_t bytes(const llama_kv_cache & cache) {
    size_t result = 0;
    for (const auto & allocation : cache.memory_breakdown()) result += allocation.second;
    return result;
}

int main() {
    setenv("LLAMA_KV_GROW", "1", 1);
    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_QWEN4EXP, llama_model_default_params()));
    auto & hp = model->hparams;
    hp = llama_hparams{};
    hp.n_layer = 1;
    hp.n_embd = hp.indexer_head_size = 32;
    hp.n_head_arr.fill(1);
    hp.n_head_kv_arr.fill(1);
    hp.n_embd_head_k_full = hp.n_embd_head_v_full = 32;
    llama_kv_cache full(*model, hp, GGML_TYPE_F32, GGML_TYPE_F32, false, false, true,
            8192, 1, 1, 0, LLAMA_SWA_TYPE_NONE, nullptr, nullptr);
    llama_memory_hybrid_idx memory(*model, GGML_TYPE_F32, GGML_TYPE_F32, false,
            8192, 1, 0, LLAMA_SWA_TYPE_NONE, GGML_TYPE_F32, GGML_TYPE_F32, 1,
            1, 0, false, true, [](int32_t) { return true; },
            [](int32_t) { return false; }, [](int32_t) { return true; });
    auto * indexer = memory.get_mem_idx();
    check(indexer && bytes(*indexer) * 2 == bytes(full));
    check(indexer->type_v() == GGML_TYPE_COUNT);
    auto ctx = ggml_context_ptr(ggml_init({32*ggml_tensor_overhead(), nullptr, true}));
    llama_kv_cache::slot_info slot;
    slot.s0 = slot.s1 = 0;
    slot.resize(1);
    slot.strm = {0};
    slot.idxs[0] = {0};
    auto * keys = indexer->get_k(ctx.get(), 0, 2048, slot);
    std::vector<float> expected(ggml_nelements(keys), 0.25f), actual(expected.size());
    ggml_backend_tensor_set(keys, expected.data(), 0, ggml_nbytes(keys));
    check(indexer->ensure_capacity(4097) && full.ensure_capacity(4097));
    check(bytes(*indexer) * 2 == bytes(full));
    keys = indexer->get_k(ctx.get(), 0, 2048, slot);
    ggml_backend_tensor_get(keys, actual.data(), 0, actual.size()*sizeof(float));
    check(actual == expected);
}
