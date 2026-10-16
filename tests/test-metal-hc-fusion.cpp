#include "ggml.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

static float sample(uint32_t & seed) {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return float(int32_t(seed >> 8) - 8388608) / 4194304.0f;
}

int main() {
    ggml_backend_load_all();
    auto backend = ggml_backend_init_by_name("MTL0", nullptr);
    if (!backend) {
        std::fprintf(stderr, "FAIL: this test requires Metal\n");
        return 1;
    }
    for (int nt : {1, 7, 1536}) {
        for (bool expose_intermediate : {false, true}) {
            const int width = nt == 7 ? 37 : 2560;
            const int streams = 4;
            auto ctx = ggml_init({4*1024*1024, nullptr, true});
            auto x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, 1, nt);
            auto w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, streams, nt);
            auto r = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, streams, nt);
            auto repeated = ggml_repeat(ctx, x, r);
            auto product = ggml_mul(ctx, repeated, w);
            auto out = ggml_add(ctx, r, product);
            if (expose_intermediate) {
                ggml_set_output(product);
            }
            auto graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, out);
            auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            uint32_t seed = 42;
            std::vector<float> vx(ggml_nelements(x)), vw(ggml_nelements(w)), vr(ggml_nelements(r));
            for (auto & v : vx) { v = sample(seed); }
            for (auto & v : vw) { v = sample(seed); }
            for (auto & v : vr) { v = sample(seed); }
            ggml_backend_tensor_set(x, vx.data(), 0, ggml_nbytes(x));
            ggml_backend_tensor_set(w, vw.data(), 0, ggml_nbytes(w));
            ggml_backend_tensor_set(r, vr.data(), 0, ggml_nbytes(r));
            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) { return 2; }
            std::vector<float> got(vr.size()), mid(expose_intermediate ? vr.size() : 0);
            ggml_backend_tensor_get(out, got.data(), 0, ggml_nbytes(out));
            if (expose_intermediate) { ggml_backend_tensor_get(product, mid.data(), 0, ggml_nbytes(product)); }
            for (size_t i = 0; i < got.size(); ++i) {
                const size_t row = i / width;
                volatile float mul = vx[(row / streams)*width + i % width] * vw[row];
                const float expected = vr[i] + mul;
                if (std::memcmp(&expected, &got[i], sizeof(float)) != 0 ||
                        (expose_intermediate && mid[i] != mul)) {
                    std::fprintf(stderr, "FAIL width=%d tokens=%d exposed=%d i=%zu got=%.9g expected=%.9g\n",
                        width, nt, expose_intermediate, i, got[i], expected);
                    return 3;
                }
            }
            const int iterations = nt == 1536 ? 20 : 100;
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < iterations; ++i) {
                if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) { return 2; }
            }
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now()-start).count()/iterations;
            std::printf("PASS exact width=%d tokens=%d exposed=%d mean_us=%.2f\n", width, nt, expose_intermediate, us);
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
        }
    }
    ggml_backend_free(backend);
    return 0;
}
