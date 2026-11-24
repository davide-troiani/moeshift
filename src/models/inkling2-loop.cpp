#include "models.h"
#include "../llama-memory-hybrid-iswa.h"
#include "../llama-moe-slot.h"
#include "../llama-moe-tensor-loader.h"
#include "../../ggml/src/ggml-metal/ggml-metal-impl.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

// Looped graph and relative attention adapted from llama.cpp #32217, #32429 and #32694.
namespace {
class inkling_relative_positions : public llm_graph_input_i {
public:
    inkling_relative_positions(ggml_context * ctx, const llama_memory_hybrid_iswa_context * memory,
            int64_t n_tokens, int32_t extent_full, int32_t extent_swa) :
        extent_full(extent_full), extent_swa(extent_swa), memory(memory) {
        full = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, memory->get_attn()->get_base()->get_n_kv(), n_tokens);
        swa = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, memory->get_attn()->get_swa()->get_n_kv(), n_tokens);
        ggml_set_input(full);
        ggml_set_input(swa);
    }

    void set_input(const llama_ubatch * ubatch) override {
        if (full->buffer) {
            memory->get_attn()->get_base()->set_input_relative_pos(full, ubatch, extent_full);
        }
        if (swa->buffer) {
            memory->get_attn()->get_swa()->set_input_relative_pos(swa, ubatch, extent_swa);
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        memory = static_cast<const llama_memory_hybrid_iswa_context *>(params.mctx);
        return full->ne[0] == memory->get_attn()->get_base()->get_n_kv() &&
                swa->ne[0] == memory->get_attn()->get_swa()->get_n_kv() &&
                full->ne[1] == params.ubatch.n_tokens;
    }

    ggml_tensor * full = nullptr;
    ggml_tensor * swa = nullptr;

private:
    const int32_t extent_full;
    const int32_t extent_swa;
    const llama_memory_hybrid_iswa_context * memory;
};
}

int llama_model_inkling2_loop::physical_layer(int il) const {
    if (il < int(core_start)) {
        return il;
    }
    const int width = core_end - core_start;
    const int end = core_start + n_passes*width;
    return il < end ? core_start + (il - core_start) % width : il - (n_passes - 1)*width;
}

int llama_model_inkling2_loop::first_slot(int ip) const {
    return ip < int(core_end) ? ip : ip + (n_passes - 1)*(core_end - core_start);
}

void llama_model_inkling2_loop::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_QK_CLIP, qk_clip);
    ml.get_key(LLM_KV_ATTN_RES_BLOCK_SIZE, block_size);
    ml.get_key(LLM_KV_SHORTCONV_L_CACHE, hparams.n_shortconv_l_cache);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_LENGTH, n_relative);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_EXTENT, relative_extent);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_CLIP, relative_clip);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_FLOOR, log_scale_floor);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_ALPHA, log_scale_alpha);
    ml.get_key(LLM_KV_VOCAB_SIZE, n_vocab_padded);
    ml.get_key(LLM_KV_VOCAB_SIZE_UNPADDED, n_vocab_unpadded);
    ml.get_key(LLM_KV_ACTIVATION_SITU_BETA, hparams.situ_beta);
    ml.get_key(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, hparams.situ_linear_beta);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE, hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping, false);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
    ml.get_key(LLM_KV_LOOP_START, core_start);
    ml.get_key(LLM_KV_LOOP_END, core_end);
    ml.get_key(LLM_KV_LOOP_COUNT, n_passes);
    ml.get_key(LLM_KV_LOOP_GATE_RANK, gate_rank);
    bool varying = false;
    ml.get_key(LLM_KV_LOOP_VARYING, varying, false);

    n_phys = hparams.n_layer;
    if (!n_phys || n_phys > LLAMA_MAX_LAYERS || !hparams.n_embd ||
            !block_size || !core_start || core_start >= core_end || core_end > n_phys ||
            core_start % block_size || core_end % block_size || n_passes < (varying ? 1u : 2u) || !gate_rank) {
        throw std::runtime_error("inkling2_loop: invalid layer, core or controller dimensions");
    }
    const uint64_t n_slots = n_phys + uint64_t(n_passes - 1)*(core_end - core_start);
    if (n_slots > LLAMA_MAX_LAYERS) {
        throw std::runtime_error("inkling2_loop: too many cache slots");
    }

    const uint32_t n_conv = hparams.n_shortconv_l_cache;
    if (!n_relative || !relative_extent || !hparams.n_swa || !log_scale_floor ||
            n_conv < 2 || n_conv > 64 || relative_extent > INT32_MAX || hparams.n_swa > INT32_MAX ||
            !n_vocab_unpadded || n_vocab_unpadded > n_vocab_padded || !hparams.n_expert_used) {
        throw std::runtime_error("inkling2_loop: invalid attention, convolution or vocabulary dimensions");
    }
    for (float value : {hparams.f_norm_rms_eps, qk_clip, relative_clip, hparams.situ_beta, hparams.situ_linear_beta}) {
        if (!std::isfinite(value) || value <= 0.0f) {
            throw std::runtime_error("inkling2_loop: normalization, clipping and activation parameters must be finite and positive");
        }
    }
    if (!std::isfinite(log_scale_alpha) || log_scale_alpha < 0.0f ||
            !std::isfinite(hparams.expert_weights_scale) || hparams.expert_weights_scale < 0.0f ||
            !std::isfinite(hparams.f_final_logit_softcapping) || hparams.f_final_logit_softcapping < 0.0f) {
        throw std::runtime_error("inkling2_loop: invalid scale or logit soft cap");
    }

    std::array<bool, LLAMA_MAX_LAYERS> swa{};
    ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, swa, n_phys);
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, expert_ff, n_phys);
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;

    // Each layer has both a KV cache and a short-convolution state.
    uint64_t n_channels_max = 0;
    for (uint32_t il = 0; il < n_phys; ++il) {
        hparams.swa_layers[il] = swa[il];
        const uint32_t n_head = hparams.n_head(il);
        const uint32_t n_head_kv = hparams.n_head_kv(il);
        const uint32_t n_head_dim = hparams.n_embd_head_k(il);
        if (!n_head || !n_head_kv || n_head % n_head_kv || !n_head_dim ||
                n_head_dim != hparams.n_embd_head_v(il) || !expert_ff[il]) {
            throw std::runtime_error("inkling2_loop: invalid head or expert dimensions");
        }
        const uint64_t n_kv_channels = uint64_t(n_head_dim)*n_head_kv;
        if (n_kv_channels > INT32_MAX) {
            throw std::runtime_error("inkling2_loop: attention state is too large");
        }
        n_channels_max = std::max(n_channels_max, 2*(n_kv_channels + hparams.n_embd));
    }
    if (n_channels_max > INT32_MAX / (n_conv - 1)) {
        throw std::runtime_error("inkling2_loop: convolution state is too large");
    }
    hparams.n_sconv_channels = n_channels_max;

    // Every visit has its own states in execution order, including repeats of the same weights.
    const auto heads = hparams.n_head_arr;
    const auto kv_heads = hparams.n_head_kv_arr;
    const auto ff = hparams.n_ff_arr;
    hparams.n_layer = n_slots;
    for (uint32_t il = core_end; il < n_slots; ++il) {
        const int ip = physical_layer(il);
        hparams.n_head_arr[il] = heads[ip];
        hparams.n_head_kv_arr[il] = kv_heads[ip];
        hparams.n_ff_arr[il] = ff[ip];
        hparams.swa_layers[il] = swa[ip];
    }
    type = LLM_TYPE_UNKNOWN;
}

void llama_model_inkling2_loop::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    if (n_vocab_unpadded != uint32_t(vocab.n_tokens())) {
        throw std::runtime_error("inkling2_loop: tokenizer size must equal the unpadded vocabulary size");
    }

    const int64_t n_rel          = n_relative;
    const int64_t n_conv         = hparams.n_shortconv_l_cache;
    const bool streaming = moe_slot_bank_enabled();
    if (streaming) {
        slot_layers.resize(n_layer);
    }
    llama_moe_tensor_loader routed(*this, ml, streaming ? &slot_layers : nullptr);
    const auto load_expert = [&](const LLM_TN_IMPL & name, const std::initializer_list<int64_t> & dims) {
        return streaming ? routed.create(name, dims, 0) : create_tensor(name, dims, 0);
    };

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,      "weight"),    {n_embd, n_vocab_padded}, 0);
    tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, 0);

    // output
    output           = create_tensor(tn(LLM_TENSOR_OUTPUT,           "weight"), {n_embd, n_vocab_padded}, 0);
    output_norm      = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,      "weight"), {n_embd}, 0);
    output_res_query = create_tensor(tn(LLM_TENSOR_OUTPUT_RES_QUERY, "weight"), {n_embd}, 0);

    weights.resize(n_phys);
    for (uint32_t i = 0; i < n_phys; ++i) {
        auto & layer = layers[i];
        auto & extra = weights[i];

        const int il = first_slot(i);
        const bool is_swa = hparams.is_swa(il);
        const int64_t n_head      = hparams.n_head(il);
        const int64_t n_head_kv   = hparams.n_head_kv(il);
        const int64_t n_embd_head = hparams.n_embd_head_k(il);
        const int64_t n_ff_exp    = expert_ff[i];
        const int64_t n_extent    = is_swa ? hparams.n_swa : relative_extent;

        // attention
        layer.wq        = create_tensor(tn(LLM_TENSOR_ATTN_Q,    "weight", i), {n_embd, n_embd_head*n_head}, 0);
        layer.wk        = create_tensor(tn(LLM_TENSOR_ATTN_K,    "weight", i), {n_embd, n_embd_head*n_head_kv}, 0);
        layer.wv        = create_tensor(tn(LLM_TENSOR_ATTN_V,    "weight", i), {n_embd, n_embd_head*n_head_kv}, 0);
        layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_OUT,  "weight", i), {n_embd_head*n_head, n_embd}, 0);
        layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_embd_head*n_head}, 0);

        extra.attn_q_clip    = create_tensor(tn(LLM_TENSOR_ATTN_Q_CLIP,    "weight", i), {n_embd_head, n_head}, 0);
        extra.attn_sink_proj = create_tensor(tn(LLM_TENSOR_ATTN_SINK_PROJ, "weight", i), {n_embd, n_head}, 0);

        // relative-position bias
        extra.attn_rel      = create_tensor(tn(LLM_TENSOR_ATTN_REL,      "weight", i), {n_embd, n_rel*n_head}, 0);
        extra.attn_rel_bias = create_tensor(tn(LLM_TENSOR_ATTN_REL_BIAS, "weight", i), {n_rel, n_extent}, 0);
        extra.attn_rel_clip = create_tensor(tn(LLM_TENSOR_ATTN_REL_CLIP, "weight", i), {n_rel, n_head}, 0);

        // short convolutions
        extra.attn_k_conv    = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV,    "weight", i), {n_conv, n_embd_head*n_head_kv}, 0);
        extra.attn_v_conv    = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV,    "weight", i), {n_conv, n_embd_head*n_head_kv}, 0);
        extra.attn_post_conv = create_tensor(tn(LLM_TENSOR_ATTN_POST_CONV, "weight", i), {n_conv, n_embd}, 0);
        extra.ffn_post_conv  = create_tensor(tn(LLM_TENSOR_FFN_POST_CONV,  "weight", i), {n_conv, n_embd}, 0);

        if (is_swa) {
            layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        } else {
            // full-attention layers: separate Q, K and V input norms, and input-dependent taps for the K and V convolutions
            extra.attn_q_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_INP_NORM,  "weight", i), {n_embd}, 0);
            extra.attn_k_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_K_INP_NORM,  "weight", i), {n_embd}, 0);
            extra.attn_v_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_V_INP_NORM,  "weight", i), {n_embd}, 0);
            extra.attn_k_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV_PROJ, "weight", i), {n_embd, n_conv*n_head_kv}, 0);
            extra.attn_v_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV_PROJ, "weight", i), {n_embd, n_conv*n_head_kv}, 0);
        }

        // block attention residuals
        if (i > 0) {
            extra.attn_res_query = create_tensor(tn(LLM_TENSOR_ATTN_RES_QUERY, "weight", i), {n_embd, is_swa ? 1 : 3}, 0);
        }
        extra.ffn_res_query = create_tensor(tn(LLM_TENSOR_FFN_RES_QUERY, "weight", i), {n_embd}, 0);

        // MoE
        layer.ffn_norm         = create_tensor(tn(LLM_TENSOR_FFN_NORM,         "weight", i), {n_embd}, 0);
        layer.ffn_gate_inp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,     "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b  = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B,  "bias",   i), {n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_gate_up_exps = load_expert(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", i), {n_embd, 2*n_ff_exp, n_expert});
        layer.ffn_up_exps_s    = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_down_exps    = load_expert(tn(LLM_TENSOR_FFN_DOWN_EXPS,    "weight", i), {n_ff_exp, n_embd, n_expert});
    }
    routed.finish();

    const int64_t rank   = gate_rank;
    const llm_tensor down[LOOP_CTRL_COUNT]   = { LLM_TENSOR_LOOP_Q_CTRL_DOWN,   LLM_TENSOR_LOOP_K_CTRL_DOWN,   LLM_TENSOR_LOOP_V_CTRL_DOWN,   LLM_TENSOR_LOOP_FFN_CTRL_DOWN   };
    const llm_tensor gain[LOOP_CTRL_COUNT]   = { LLM_TENSOR_LOOP_Q_CTRL_GAIN,   LLM_TENSOR_LOOP_K_CTRL_GAIN,   LLM_TENSOR_LOOP_V_CTRL_GAIN,   LLM_TENSOR_LOOP_FFN_CTRL_GAIN   };
    const llm_tensor forget[LOOP_CTRL_COUNT] = { LLM_TENSOR_LOOP_Q_CTRL_FORGET, LLM_TENSOR_LOOP_K_CTRL_FORGET, LLM_TENSOR_LOOP_V_CTRL_FORGET, LLM_TENSOR_LOOP_FFN_CTRL_FORGET };

    loop_layers.resize(core_end - core_start);
    for (uint32_t il = core_start; il < core_end; ++il) {
        auto & l = loop_layers[il - core_start];
        // the q, k, v and ffn message queries
        l.msg_query = create_tensor(tn(LLM_TENSOR_LOOP_MSG_QUERY, "weight", il), {n_embd, LOOP_CTRL_COUNT}, 0);
        for (int c = 0; c < LOOP_CTRL_COUNT; ++c) {
            l.ctrl_down[c]   = create_tensor(tn(down[c],   "weight", il), {2*n_embd, rank}, 0);
            l.ctrl_gain[c]   = create_tensor(tn(gain[c],   "weight", il), {rank, n_embd}, 0);
            l.ctrl_forget[c] = create_tensor(tn(forget[c], "weight", il), {rank, n_embd}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_inkling2_loop::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// scale each row down to an L2 norm of at most threshold, after the optional per-channel scale
ggml_tensor * llama_model_inkling2_loop::graph::clip(ggml_tensor * cur, ggml_tensor * scale, float threshold) {
    ggml_tensor * input = cur;
    if (scale) {
        cur = ggml_mul(ctx0, cur, scale);
    }

    ggml_tensor * squares = ggml_sqr(ctx0, cur);
    ggml_tensor * norm = ggml_sqrt(ctx0, ggml_sum_rows(ctx0, squares));
    norm = ggml_scale_bias(ctx0, norm, 1.0f, 1e-6f);

    ggml_tensor * factor = ggml_div(ctx0, ggml_fill(ctx0, norm, threshold), norm);

    ggml_tensor * out = ggml_mul(ctx0, cur, ggml_clamp(ctx0, factor, 0.0f, 1.0f));
    if (moe_decode_phase && n_tokens == 1) {
        (scale ? cur : squares)->op_params[GGML_INKLING_FUSION_PARAM] = GGML_INKLING_CLIP;
        // The fused kernel reads the original inputs when the final MUL runs.
        out->src[2] = input;
        out->src[3] = scale;
    }
    return out;
}

ggml_tensor * llama_model_inkling2_loop::graph::norm_residual(ggml_tensor * cur) {
    return ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens), 1e-6f);
}

ggml_tensor * llama_model_inkling2_loop::graph::residual_states(ggml_tensor * blocks, ggml_tensor * cur) {
    ggml_tensor * state = norm_residual(cur);
    return blocks ? ggml_concat(ctx0, blocks, state, 1) : state;
}

// attention over the depth: mix the normalized residuals with learned queries
ggml_tensor * llama_model_inkling2_loop::graph::mix(ggml_tensor * states, ggml_tensor * query, ggml_tensor * transposed) {
    ggml_tensor * queries = ggml_repeat(ctx0, query, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, query->ne[1], n_tokens));

    ggml_tensor * scores = ggml_mul_mat(ctx0, states, queries);
    scores = ggml_soft_max_ext(ctx0, scores, nullptr, 1.0f/sqrtf(n_embd), 0.0f);

    if (!transposed) {
        transposed = ggml_cont(ctx0, ggml_transpose(ctx0, states));
    }
    return ggml_mul_mat(ctx0, transposed, scores);
}

// causal short convolution with a residual; dynamic adds input-dependent taps per head
// the last n_conv - 1 inputs of each sequence live in the recurrent state at channel offset `offset`
ggml_tensor * llama_model_inkling2_loop::graph::convolve(
        ggml_tensor * cur,
        ggml_tensor * weight,
        ggml_tensor * dynamic,
        ggml_tensor * state,
        ggml_tensor * history,
        int64_t offset) {
    const int64_t n_conv       = hparams.n_shortconv_l_cache;
    const int64_t n_channels   = cur->ne[0];
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t n_seqs       = ubatch.n_seqs;

    ggml_tensor * past = ggml_view_3d(ctx0, history, n_conv - 1, n_channels, n_seqs, history->nb[1], history->nb[2], offset*history->nb[1]);

    ggml_tensor * x = ggml_transpose(ctx0, ggml_reshape_3d(ctx0, cur, n_channels, n_seq_tokens, n_seqs));
    x = ggml_concat(ctx0, past, x, 0);

    const int64_t n_history = n_conv - 1;
    ggml_tensor * tail = ggml_view_3d(ctx0, x, n_history, n_channels, n_seqs,
            x->nb[1], x->nb[2], n_seq_tokens*x->nb[0]);
    const size_t state_offset = recr->mctx->get_head()*state->nb[1] + offset*n_history*state->nb[0];
    ggml_tensor * cell = ggml_view_2d(ctx0, state, n_history*n_channels, n_seqs, state->nb[1], state_offset);
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, tail, cell));

    ggml_tensor * out = ggml_ssm_conv(ctx0, x, weight);
    out = ggml_reshape_2d(ctx0, out, n_channels, n_tokens);

    if (dynamic) {
        const int64_t n_heads = dynamic->ne[0]/n_conv;

        dynamic = ggml_reshape_3d(ctx0, dynamic, n_conv, n_heads, n_tokens);

        for (int64_t lag = 0; lag < n_conv; ++lag) {
            ggml_tensor * tap    = ggml_view_3d(ctx0, dynamic, 1, n_heads, n_tokens, dynamic->nb[1], dynamic->nb[2], lag*dynamic->nb[0]);
            ggml_tensor * window = ggml_view_3d(ctx0, x, n_seq_tokens, n_channels, n_seqs, x->nb[1], x->nb[2], (n_conv - 1 - lag)*x->nb[0]);

            window = ggml_cont(ctx0, ggml_transpose(ctx0, window));
            window = ggml_reshape_3d(ctx0, window, n_channels/n_heads, n_heads, n_tokens);

            ggml_tensor * term = ggml_reshape_2d(ctx0, ggml_mul(ctx0, window, tap), n_channels, n_tokens);

            out = ggml_add(ctx0, out, term);
        }
    }

    return ggml_scale(ctx0, ggml_add(ctx0, cur, out), 1.0f/sqrtf((dynamic ? 2 : 1)*n_conv + 1));
}

// mix a cross-pass message into a looped layer input: (1 - forget)*cur + gain*message, both gated on cur and message
ggml_tensor * llama_model_inkling2_loop::graph::control(ggml_tensor * cur, ggml_tensor * message, const loop_layer & loop, int ctrl) {
    ggml_tensor * gate = ggml_concat(ctx0, ggml_rms_norm(ctx0, cur, 1e-6f), ggml_rms_norm(ctx0, message, 1e-6f), 0);
    gate = build_lora_mm(loop.ctrl_down[ctrl], gate);

    ggml_tensor * gain   = ggml_tanh(ctx0, build_lora_mm(loop.ctrl_gain[ctrl],   gate));
    ggml_tensor * forget = ggml_tanh(ctx0, build_lora_mm(loop.ctrl_forget[ctrl], gate));

    return ggml_add(ctx0, ggml_sub(ctx0, cur, ggml_mul(ctx0, forget, cur)), ggml_mul(ctx0, gain, message));
}

ggml_tensor * llama_model_inkling2_loop::graph::build_inputs() {
    auto * memory_input = build_inp_mem_hybrid_iswa();
    attn = memory_input->get_attn();
    recr = memory_input->get_recr();

    const auto * memory = static_cast<const llama_memory_hybrid_iswa_context *>(mctx);
    auto positions = std::make_unique<inkling_relative_positions>(ctx0, memory, n_tokens, inkling.relative_extent, hparams.n_swa);
    rel_pos[0] = positions->full;
    rel_pos[1] = positions->swa;
    res->add_input(std::move(positions));

    ggml_tensor * inp_pos = build_inp_pos();
    inp_out_ids = build_inp_out_ids();

    // log-scaled positions: 1 + alpha*log(max(1, (pos + 1)/floor)), multiplies q and r of the full-attention layers
    tau = ggml_scale_bias(ctx0, ggml_cast(ctx0, inp_pos, GGML_TYPE_F32), 1.0f, 1.0f);
    tau = ggml_log(ctx0, ggml_clamp(ctx0, ggml_scale(ctx0, tau, 1.0f/inkling.log_scale_floor), 1.0f, INFINITY));
    tau = ggml_reshape_3d(ctx0, ggml_scale_bias(ctx0, tau, inkling.log_scale_alpha, 1.0f), 1, 1, n_tokens);

    ggml_tensor * cur = build_inp_embd(inkling.tok_embd);

    // important: do not normalize raw embeddings input (i.e. encoded image embeddings)
    if (ubatch.token) {
        cur = build_norm(cur, inkling.tok_norm, nullptr, LLM_NORM_RMS, -1);
    }

    return cur;
}

ggml_tensor * llama_model_inkling2_loop::graph::build_layer(int il, int ip, ggml_tensor * cur, ggml_tensor * & blocks,
        ggml_tensor * messages, const loop_layer * loop) {
    GGML_ASSERT((messages == nullptr) == (loop == nullptr));

    const auto & layer = inkling.layers[ip];
    const auto & extra = inkling.weights[ip];

    const bool    is_swa       = hparams.is_swa(il);
    const int64_t n_head       = hparams.n_head(il);
    const int64_t n_head_kv    = hparams.n_head_kv(il);
    const int64_t n_embd_head  = hparams.n_embd_head_k(il);
    const int64_t n_embd_k_gqa = n_embd_head*n_head_kv;

    // the message of controller ctrl, [n_embd, n_tokens]
    auto message = [&](int ctrl) {
        return ggml_cont(ctx0, ggml_view_2d(ctx0, messages, n_embd, n_tokens, messages->nb[2], ctrl*messages->nb[1]));
    };

    ggml_tensor * state   = recr->mctx->get_r_l(il);
    ggml_tensor * history = build_rs(recr, state, hparams.n_embd_r(), ubatch.n_seqs);
    history = ggml_reshape_3d(ctx0, history, hparams.n_shortconv_l_cache - 1, hparams.n_sconv_channels, ubatch.n_seqs);

    ggml_tensor * inpSA = cur;

    // attention inputs: SWA layers mix one, full-attention layers one each for Q, K and V
    ggml_tensor * inputs[3] = {cur, cur, cur};
    if (extra.attn_res_query) {
        ggml_tensor * mixed = mix(residual_states(blocks, cur), extra.attn_res_query);
        if (is_swa) {
            inputs[0] = inputs[1] = inputs[2] = ggml_reshape_2d(ctx0, mixed, n_embd, n_tokens);
        } else {
            for (int j = 0; j < 3; ++j) {
                inputs[j] = ggml_cont(ctx0, ggml_view_2d(ctx0, mixed, n_embd, n_tokens, mixed->nb[2], j*mixed->nb[1]));
            }
        }
    }

    if (messages) {
        // the messages make the Q, K and V inputs of a SWA layer differ
        for (int j = 0; j < 3; ++j) {
            inputs[j] = control(inputs[j], message(LOOP_CTRL_Q + j), *loop, LOOP_CTRL_Q + j);
        }
    }

    if (is_swa && messages) {
        for (int j = 0; j < 3; ++j) {
            inputs[j] = build_norm(inputs[j], layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        }
    } else if (is_swa) {
        inputs[0] = build_norm(inputs[0], layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        inputs[1] = inputs[2] = inputs[0];
    } else {
        inputs[0] = build_norm(inputs[0], extra.attn_q_inp_norm, nullptr, LLM_NORM_RMS, il);
        inputs[1] = build_norm(inputs[1], extra.attn_k_inp_norm, nullptr, LLM_NORM_RMS, il);
        inputs[2] = build_norm(inputs[2], extra.attn_v_inp_norm, nullptr, LLM_NORM_RMS, il);
    }

    // self-attention
    {
        ggml_tensor * Qcur = build_lora_mm(layer.wq, inputs[0], layer.wq_s);
        ggml_tensor * Kcur = convolve(build_lora_mm(layer.wk, inputs[1], layer.wk_s), extra.attn_k_conv,
                is_swa ? nullptr : build_lora_mm(extra.attn_k_conv_proj, inputs[1]), state, history, 0);
        ggml_tensor * Vcur = convolve(build_lora_mm(layer.wv, inputs[2], layer.wv_s), extra.attn_v_conv,
                is_swa ? nullptr : build_lora_mm(extra.attn_v_conv_proj, inputs[2]), state, history, n_embd_k_gqa);
        ggml_tensor * Rcur = build_lora_mm(extra.attn_rel, inputs[0]);

        Qcur = clip(ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens), extra.attn_q_clip, inkling.qk_clip);
        Kcur = clip(ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), nullptr,           inkling.qk_clip);
        Vcur = ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens), hparams.f_norm_rms_eps);
        Rcur = clip(ggml_reshape_3d(ctx0, Rcur, inkling.n_relative, n_head, n_tokens), extra.attn_rel_clip, inkling.relative_clip);

        if (!is_swa) {
            Qcur = ggml_mul(ctx0, Qcur, tau);
            Rcur = ggml_mul(ctx0, Rcur, tau);
        }

        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);
        cb(Rcur, "Rcur", il);

        ggml_tensor * sinks = build_lora_mm(extra.attn_sink_proj, inputs[0]);

        // project r on the bias table and look up the projected bias by distance
        ggml_tensor * table = ggml_mul_mat(ctx0, ggml_pad(ctx0, extra.attn_rel_bias, 0, 1, 0, 0), Rcur);

        // the sink logit t is per query: exp(s - t)/(1 + sum(exp(s - t))) = exp(s)/(exp(t) + sum(exp(s))), with a zero sink
        table = ggml_sub(ctx0, table, ggml_reshape_3d(ctx0, sinks, 1, n_head, n_tokens));

        sinks = ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_head), 0.0f);

        cur = build_attn(attn,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, sinks, nullptr, 1.0f/sqrtf(n_embd_head), il, table, rel_pos[is_swa]);

        cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, build_lora_mm(layer.wqkv_gate, inputs[0], layer.wqkv_gate_s)));
        cur = convolve(build_lora_mm(layer.wo, cur, layer.wo_s), extra.attn_post_conv, nullptr, state, history, 2*n_embd_k_gqa);
        cb(cur, "attn_out", il);
    }

    // the first layer of each block starts a new residual stream, the earlier one joins the mixed blocks
    if (ip % inkling.block_size == 0) {
        ggml_tensor * state = norm_residual(inpSA);
        blocks = blocks ? ggml_concat(ctx0, state, blocks, 1) : state;
    } else {
        cur = ggml_add(ctx0, inpSA, cur);
    }

    ggml_tensor * ffn_inp = cur;
    cb(ffn_inp, "ffn_inp", il);

    // MoE
    cur = ggml_reshape_2d(ctx0, mix(residual_states(blocks, cur), extra.ffn_res_query), n_embd, n_tokens);
    if (messages) {
        cur = control(cur, message(LOOP_CTRL_FFN), *loop, LOOP_CTRL_FFN);
    }
    cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);

    if (moe_slots) {
        ggml_tensor * prediction = nullptr;
        if (moe_decode_phase && moe_slots->lookahead_prefetch_enabled() && il + 1 < n_layer) {
            const auto & next = inkling.layers[inkling.physical_layer(il + 1)];
            auto * scores = ggml_sigmoid(ctx0, build_lora_mm(next.ffn_gate_inp, cur));
            if (next.ffn_exp_probs_b) {
                scores = ggml_add(ctx0, scores, next.ffn_exp_probs_b);
            }
            prediction = ggml_cont(ctx0, ggml_argsort_top_k(ctx0, scores, moe_slots->lookahead_prefetch_depth()));
            cb(prediction, "prefetch_topk", il);
            ggml_build_forward_expand(gf, prediction);
        }
        moe_slots->set_lookahead_pred(prediction);
    }

    const auto & experts = inkling.slot_layers.empty() ? layer : inkling.slot_layers[il];
    auto * gate_up = moe_decode_phase && experts.ffn_gate_up_exps_dec ? experts.ffn_gate_up_exps_dec : experts.ffn_gate_up_exps;
    auto * down = moe_decode_phase && experts.ffn_down_exps_dec ? experts.ffn_down_exps_dec : experts.ffn_down_exps;
    cur = build_moe_ffn(cur,
            layer.ffn_gate_inp,
            nullptr,
            nullptr,
            down,
            layer.ffn_exp_probs_b,
            hparams.n_expert,
            hparams.n_expert_used,
            LLM_FFN_SITU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID,
            il,
            nullptr,
            gate_up,
            layer.ffn_up_exps_s, nullptr, layer.ffn_down_exps_s);

    cur = convolve(cur, extra.ffn_post_conv, nullptr, state, history, 2*n_embd_k_gqa + n_embd);
    cb(cur, "ffn_out", il);

    cur = ggml_add(ctx0, ffn_inp, cur);
    cb(cur, "l_out", il);

    return cur;
}

void llama_model_inkling2_loop::graph::build_output(ggml_tensor * cur, ggml_tensor * blocks) {
    cur = ggml_reshape_2d(ctx0, mix(residual_states(blocks, cur), inkling.output_res_query), n_embd, n_tokens);
    cur = build_norm(cur, inkling.output_norm, nullptr, LLM_NORM_RMS, -1);

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(inkling.output, cur, inkling.output_s);
    if (cur->op == GGML_OP_MUL_MAT) {
        ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
    }

    if (hparams.f_final_logit_softcapping != 0.0f) {
        const float cap = hparams.f_final_logit_softcapping;

        cur = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/cap)), cap);
    }

    // drop the padded vocabulary
    cur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, inkling.n_vocab_unpadded, cur->ne[1], cur->nb[1], 0));

    cb(cur, "result_output", -1);
    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}

llama_model_inkling2_loop::graph::graph(const llama_model_inkling2_loop & inkling, const llm_graph_params & params) :
    llm_graph_context(params), inkling(inkling) {
    const int start = inkling.core_start;
    const int end   = inkling.core_end;

    auto * cur = build_inputs();
    ggml_tensor * blocks = nullptr;
    int il = 0;
    for (; il < start; ++il) {
        cur = build_layer(il, il, cur, blocks);
    }

    // every pass starts from the prelude output
    auto * prelude = cur;
    auto * prelude_blocks = blocks;
    for (uint32_t pass = 0; pass < inkling.n_passes; ++pass) {
        // the bank of the previous pass: its blocks after the first core block, and its output
        ggml_tensor * bank = nullptr;
        ggml_tensor * bank_t = nullptr;
        if (pass > 0) {
            ggml_tensor * bank_blocks = nullptr;
            const int64_t n_bank_blocks = blocks->ne[1] - prelude_blocks->ne[1] - 1;
            if (n_bank_blocks > 0) {
                bank_blocks = ggml_view_3d(ctx0, blocks, n_embd, n_bank_blocks, n_tokens, blocks->nb[1], blocks->nb[2], 0);
            }
            bank = residual_states(bank_blocks, cur);
            bank_t = ggml_cont(ctx0, ggml_transpose(ctx0, bank));
            cur = prelude;
            blocks = prelude_blocks;
        }
        for (int ip = start; ip < end; ++ip, ++il) {
            if (pass == 0) {
                cur = build_layer(il, ip, cur, blocks);
                continue;
            }
            const auto & loop = inkling.loop_layers[ip - start];
            cur = build_layer(il, ip, cur, blocks, mix(bank, loop.msg_query, bank_t), &loop);
        }
    }

    for (int ip = end; ip < int(inkling.n_phys); ++ip, ++il) {
        cur = build_layer(il, ip, cur, blocks);
    }
    build_output(cur, blocks);
}
