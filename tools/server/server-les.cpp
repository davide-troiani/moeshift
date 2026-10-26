#ifdef LLAMA_LES_METAL_KEEPALIVE
#include "gpu-keepalive.h"
#endif

#include "server-context.h"
#include "server-http.h"

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <atomic>
#include <algorithm>
#include <chrono>
#include <csignal>
#include <iostream>
#include <thread>

static volatile std::sig_atomic_t stop_requested = 0;

static void handle_signal(int) {
    stop_requested = 1;
}

int main(int argc, char ** argv) {
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    common_params params;

    common_init();

    if (!common_params_parse(
            argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    if (params.n_parallel != 1 || std::any_of(params.speculative.types.begin(), params.speculative.types.end(),
            [](common_speculative_type type) { return type != COMMON_SPECULATIVE_TYPE_NONE; }) ||
        params.moe_experts.empty() || params.les_decode_bank >= params.les_prefill_bank) {
        std::cerr << "Adaptive LES requires parallel=1, spec-type=none, MoE experts, and decode bank < prefill bank\n";
        return 1;
    }
    params.n_batch = params.n_ubatch = params.les_prefill_max_tokens;
    params.moe_slot_bank = params.les_prefill_bank;
    setenv("LLAMA_LES_DECODE_BANK", std::to_string(params.les_decode_bank).c_str(), 1);
    std::cerr << "LES adaptive: prefill-max=" << params.n_ubatch
              << " shared-bank=" << params.les_prefill_bank << " -> " << params.les_decode_bank << "\n";

    // Use a single inference slot by default.
    if (params.n_parallel < 0) {
        params.n_parallel = 1;
    }

#ifdef LLAMA_LES_METAL_KEEPALIVE
    gpu_keepalive keepalive;
    if (!keepalive.ready) {
        return 1;
    }
#endif

    llama_backend_init();
    llama_numa_init(params.numa);

    int result = 0;

    {
        server_context ctx;
        server_routes routes(params, ctx);
        server_http_context http;

        if (!http.init(params)) {
            std::cerr << "HTTP initialization failed\n";
            result = 1;
        } else {
            // OpenAI-compatible endpoints.
            http.get(
                "/health",
                routes.get_health
            );

            http.get(
                "/v1/models",
                routes.get_models
            );

            http.post(
                "/v1/chat/completions",
                routes.post_chat_completions
            );

            http.post(
                "/v1/completions",
                routes.post_completions_oai
            );

            // Load the model through the existing
            // llama.cpp inference backend.
            if (!ctx.load_model(params)) {
                std::cerr << "Model loading failed\n";
                result = 1;
            } else {
                routes.update_meta(ctx);

                if (!http.start()) {
                    std::cerr << "HTTP server startup failed\n";
                    result = 1;
                } else {
                    http.is_ready.store(true);

                    std::cout
                        << "OpenAI-compatible API listening on "
                        << http.listening_address
                        << "/v1\n";

                    std::thread inference([&ctx]() {
                        ctx.start_loop();
                    });

                    while (!stop_requested) {
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(200)
                        );
                    }

                    ctx.terminate();
                    inference.join();

                    http.stop();
                    if (http.thread.joinable()) {
                        http.thread.join();
                    }
                }
            }
        }
    }

    llama_backend_free();

    return result;
}
