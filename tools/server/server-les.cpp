#ifdef LLAMA_LES_METAL_KEEPALIVE
#include "gpu-keepalive.h"
#endif

#include "server-context.h"
#include "server-http.h"
#include "server-les-router.h"

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <atomic>
#include <algorithm>
#include <chrono>
#include <csignal>
#include <iostream>
#include <filesystem>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif
#include <thread>

static volatile std::sig_atomic_t stop_requested = 0;

static void handle_signal(int) {
    stop_requested = 1;
}

int main(int argc, char ** argv) {
    common_log_set_application(common_log_main(), "moeshift");
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    if (argc < 2 || std::string(argv[1]) != "--worker") {
        if (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
            std::cout << "Usage: moeshift [--config FILE] [--model GGUF] [--moe-experts DIR] [--port N]\n"
                      << "Profile options override the configuration file; inference parallelism is always 1.\n";
            return 0;
        }
        std::string config = "moeshift.json";
        int first = 1;
        if (argc > 1 && std::string(argv[1]) == "--config") {
            if (argc < 3) { std::cerr << "moeshift: --config requires a file\n"; return 1; }
            config = argv[2]; first = 3;
        } else if (!std::filesystem::exists(config)) {
            const char * home = std::getenv("HOME");
            const char * xdg = std::getenv("XDG_CONFIG_HOME");
            const auto user_config = xdg && *xdg ? std::filesystem::path(xdg) / "moeshift/config.json"
                : std::filesystem::path(home ? home : "") / ".config/moeshift/config.json";
            if (std::filesystem::exists(user_config)) config = user_config.string();
            else {
#ifdef __APPLE__
            uint32_t size = 0;
            _NSGetExecutablePath(nullptr, &size);
            std::vector<char> path(size);
            if (_NSGetExecutablePath(path.data(), &size) == 0) {
                auto prefix = std::filesystem::canonical(path.data()).parent_path().parent_path();
                auto installed = prefix / "share/moeshift/moeshift.json";
                config = installed.string();
            }
#endif
            }
        }
        return run_les_router(config.c_str(), stop_requested, argc - first, argv + first);
    }
    --argc;
    ++argv;

    common_params params;

    common_init();

    if (!common_params_parse(
            argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    // LES always serves one inference at a time with the experts page cache enabled.
    params.n_parallel = 1;
    params.hostname = "127.0.0.1";
    setenv("LLAMA_MOE_F_NOCACHE", "0", 1);

    if (std::any_of(params.speculative.types.begin(), params.speculative.types.end(),
            [](common_speculative_type type) { return type != COMMON_SPECULATIVE_TYPE_NONE; }) ||
        params.moe_experts.empty() || params.les_decode_bank >= params.les_prefill_bank) {
        std::cerr << "moeshift: adaptive expert streaming requires spec-type=none, MoE experts, and decode bank < prefill bank\n";
        return 1;
    }
    // Enable growing KV buffers for LES; other clients retain the original allocator.
    setenv("LLAMA_KV_GROW", "1", 1);
    params.n_batch = params.n_ubatch = params.les_prefill_max_tokens;
    params.moe_slot_bank = params.les_prefill_bank;
    setenv("LLAMA_LES_DECODE_BANK", std::to_string(params.les_decode_bank).c_str(), 1);
    const char * moe_quiet = std::getenv("LLAMA_MOE_QUIET");
    if (!moe_quiet || std::string(moe_quiet) != "1") {
        std::cerr << "moeshift: adaptive: prefill-max=" << params.n_ubatch
                  << " shared-bank=" << params.les_prefill_bank << " -> " << params.les_decode_bank << "\n";
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
            std::cerr << "moeshift: HTTP initialization failed\n";
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
                std::cerr << "moeshift: Model loading failed\n";
                result = 1;
            } else {
                routes.update_meta(ctx);

                if (!http.start()) {
                    std::cerr << "moeshift: HTTP server startup failed\n";
                    result = 1;
                } else {
                    http.is_ready.store(true);

                    std::cout
                        << "moeshift: OpenAI-compatible API listening on "
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
