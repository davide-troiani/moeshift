#pragma once
#include "llama-model.h"
#include <map>

// Shared prefill banks and per-layer decode windows, grouped by family and quantization.
class llama_moe_tensor_loader {
public:
    llama_moe_tensor_loader(llama_model_base & model, llama_model_loader & ml);
    ggml_tensor * create(const LLM_TN_IMPL & name, const std::initializer_list<int64_t> & dims, int flags);
    void create_gate_up(llama_layer & layer, int il, int64_t n_ff_exp, int flags);
    void finish();
private:
    llama_model_base & model;
    llama_model_loader & ml;
    const LLM_TN tn;
    int64_t dec_c, k_pf, prefill_cap;
    bool dec_enabled;
    std::vector<int32_t> layer_c;
    std::map<std::pair<ggml_type, size_t>, std::vector<int>> groups[4];
    std::map<std::string, int64_t> group_slots;
    int64_t width_of(int il) const {
        return dec_enabled && il >= 0 && il < (int) layer_c.size() ? layer_c[il] + k_pf : 0;
    }
    int64_t slots_for(const LLM_TN_IMPL & name) const {
        const auto it = group_slots.find(name.str());
        return it == group_slots.end() ? prefill_cap : it->second;
    }
};
