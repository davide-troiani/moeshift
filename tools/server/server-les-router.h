#pragma once

#include <csignal>

int run_les_router(const char * config_path, const volatile std::sig_atomic_t & stop_requested, int argc = 0, char ** argv = nullptr);
