#include "ggml.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-metal/ggml-metal-impl.h"
#include <stdexcept>
#include <vector>

int main() {
    ggml_backend_load_all();
    auto backend = ggml_backend_init_by_name("MTL0", nullptr);
    if (!backend) throw std::runtime_error("Metal is required");
    for (int rows : {1, 3}) for (int mask = 0; mask < 4; ++mask) {
        auto * ctx = ggml_init({1024*1024, nullptr, true});
        auto * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 1, 2);
        auto * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 1, rows);
        auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, rows);
        auto * resident = ggml_mul_mat_id(ctx, weights, input, ids);
        auto * missing = ggml_mul_mat_id(ctx, weights, input, ids);
        resident->op_params[GGML_MOE_PHASE_PARAM] = GGML_MOE_PHASE_RESIDENT;
        missing->op_params[GGML_MOE_PHASE_PARAM] = GGML_MOE_PHASE_MISSING;
        auto * sum = ggml_add(ctx, resident, missing);
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, sum);
        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        std::vector<float> w(64, 1), x(32*rows, 1), result(2*rows), ready(2*rows), late(2*rows);
        std::fill(w.begin()+32, w.end(), 2);
        std::vector<int32_t> slots(2*rows);
        for (int r=0; r<rows; ++r) for (int e=0; e<2; ++e)
            slots[2*r+e] = mask & (1<<e) ? -e-1 : e;
        ggml_backend_tensor_set(weights, w.data(), 0, ggml_nbytes(weights));
        ggml_backend_tensor_set(input, x.data(), 0, ggml_nbytes(input));
        ggml_backend_tensor_set(ids, slots.data(), 0, ggml_nbytes(ids));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("Metal phase computation failed");
        ggml_backend_tensor_get(sum, result.data(), 0, ggml_nbytes(sum));
        ggml_backend_tensor_get(resident, ready.data(), 0, ggml_nbytes(resident));
        ggml_backend_tensor_get(missing, late.data(), 0, ggml_nbytes(missing));
        for (int i=0; i<2*rows; ++i) {
            const float expected = 32.0f*(i%2+1);
            const bool pending = slots[i]<0;
            if (result[i]!=expected || ready[i]!=(pending?0:expected) || late[i]!=(pending?expected:0))
                throw std::runtime_error("resident/missing phase selection changed the expert result");
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    ggml_backend_free(backend);
}
