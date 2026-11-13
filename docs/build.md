# Build MoEShift from source

Requires Apple Silicon, Apple Command Line Tools and CMake. Python 3.10+ is
needed for expert preparation and the Python checks. Use the existing GGUF
package dependencies; see `gguf-py/pyproject.toml`.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_NATIVE=ON \
  -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_APP=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=ON \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build build --target moeshift -j 8
cmake --install build --component moeshift
```

This installs the executable, default configuration and expert preparation
sources. Homebrew installs the wizard's Python environment automatically.
For a source install, activate a Python environment with the GGUF dependencies
before the first run. Run `~/.local/bin/moeshift` to choose a model and start
the server. Future launches use the saved configuration directly.

For preparation from the source checkout:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install ./gguf-py
.venv/bin/python tools/moe/experts.py extract --model "$MODEL" --out-dir "$EXPERTS"
.venv/bin/python tools/moe/experts.py verify --model "$MODEL" --experts "$EXPERTS"
```

`MODEL` must name the first GGUF shard and `EXPERTS` a new output directory.
Keep the original shards alongside the generated files.

## Focused native checks

```sh
cmake --build build --target test-les-protected-cache test-les-active-ranges \
  test-metal-moe-phases test-indexer-keys test-kv-grow test-kv-state-compat \
  test-metal-hc-fusion test-ssd-rows -j 8
ctest --test-dir build -R '^(test-les-protected-cache|test-les-active-ranges|test-metal-moe-phases|test-indexer-keys|test-kv-grow|test-metal-hc-fusion|test-ssd-rows|test-kv-state-compat)$' --output-on-failure
PYTHONDONTWRITEBYTECODE=1 python3 tests/test-moeshift-router.py build/bin/moeshift
.venv/bin/python tests/test-moeshift-onboarding.py
.venv/bin/python tests/test-moeshift-inference.py build/bin
```

Real-model validation is separate from these focused checks. The parent-lifetime
check takes the executable and a model configuration:

```sh
python3 tests/test-moeshift-parent.py build/bin/moeshift "$HOME/.config/moeshift/config.json"
```

## Additional tools

See [supported models and tools](supported-architectures.md) to choose a tool
for your task. Build the additional executables with:

```sh
cmake --build build --target llama-cli llama-completion llama-mtmd-cli \
  llama-tts-server -j 8
```

For embedding and diffusion examples, configure with the existing
`-DLLAMA_BUILD_EXAMPLES=ON`, then build `llama-embedding` and `llama-diffusion`.
To check CPU-only builds on Apple Silicon, use a separate build directory with
`-DGGML_METAL=OFF`. SSD expert streaming continues to require Metal.

The onboarding and synthetic inference checks require the gguf-py Python dependencies. The
inference check creates temporary synthetic dense Llama, Qwen3 MoE and Mamba
GGUFs, checks chat/completions with CPU and requested GPU offload, and removes
the files when done. Pass a reference `llama-completion` executable as a second
argument to compare deterministic output against upstream. Synthetic checks
verify execution, not trained-model quality or performance.

`tests/test-kv-state-compat.hex` contains one baseline M-RoPE cell (position 5,
spatial x=7/y=11, sequence 0) and F32 K/V rows. It was accepted and reproduced
byte-for-byte by upstream `96bbf61`; the regression test covers whole-context
and single-sequence restores and the existing Qwen4Exp extended layout.

