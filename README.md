# MoEShift

Local MoE inference for Mac with Apple Silicon, optimized for Qwen 3.8 Flash Next.
Routed experts are read from SSD into a bounded Metal cache. The server exposes
an OpenAI-compatible HTTP API on localhost, with one inference request at a time.

Repository: [davide-troiani/moeshift](https://github.com/davide-troiani/moeshift).
Development remains private until release preparation is complete.

## Build and install

Requirements: Apple Silicon Mac, Apple Command Line Tools, and CMake.
The commands below build a static executable with embedded Metal shaders.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DLLAMA_OPENSSL=OFF \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build build --target moeshift -j 8
cmake --install build --component moeshift
```

The installed executable is `~/.local/bin/moeshift`. Add `~/.local/bin` to your
`PATH` if needed. Inference does not require Python or a running helper service.

## Prepare model files

Start with the original language-model GGUF, including **all shards** if split.
Qwen prediction-only/MTP GGUFs and appended prediction blocks are unsupported.
Keep the GGUF files: extraction creates additional files and does not rewrite,
strip or requantize the originals.

Expert preparation uses the repository's existing Python GGUF package, requiring
Python 3.10+ and its declared dependencies:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install ./gguf-py
```

Choose a **new, empty** output directory. For a split GGUF, pass the first shard;
the remaining shards must keep their original names beside it.

```sh
MODEL="/absolute/path/model-00001-of-00002.gguf"
EXPERTS="/absolute/path/experts-new"
.venv/bin/python tools/moe/experts.py extract --model "$MODEL" --out-dir "$EXPERTS"
.venv/bin/python tools/moe/experts.py verify --model "$MODEL" --experts "$EXPERTS"
```

The output contains `manifest.json` and `layer_NNN.bin` files. Each layer file
stores the routed experts in expert-major order; the manifest records tensor
names, quantization, shapes, byte offsets and strides. Reserve additional disk
space approximately equal to the routed-expert tensor bytes, while retaining
the original GGUF shards.

**A usable model is one matched set:** the original GGUF shards, the manifest
and every `.bin` referenced by that manifest. Do not combine files from different
models, quantizations or extraction runs. Matching sizes and shapes alone do
not prove that weights match: run the full `verify` command above, which compares
metadata and routed-expert bytes against the originals. `--metadata-only` skips
the byte comparison and is only a quick structural check.

If extraction is interrupted, retry into another empty directory and verify it
before use. `--force` replaces files individually; it does not make the whole
output directory an atomic transaction. Keep incomplete output out of the
server configuration. Generated manifests include absolute source paths for
provenance; keep generated weights and manifests local rather than committing
them as examples.

## Configure and run

Copy the default profile, then set `args.model` to the first GGUF shard and
`args.moe-experts` to the verified expert directory:

```sh
mkdir -p ~/.config/moeshift
cp -n configs/moeshift.json ~/.config/moeshift/config.json
~/.local/bin/moeshift
```

Configuration lookup: `moeshift.json` in the current directory, then
`$XDG_CONFIG_HOME/moeshift/config.json` (or `~/.config/moeshift/config.json`), then
the installed default. Use `--config FILE` to choose a file explicitly. Paths in
profiles resolve relative to that file; absolute model paths are convenient.

```sh
moeshift --config configs/moeshift.json --model "$MODEL" --moe-experts "$EXPERTS" --port 8080
```

CLI profile overrides apply to every configured model. The JSON profile is the
single source of runtime tuning: shell variables starting with `LLAMA_`, `GGML_`,
`LES_` and `MB_` are not inherited by inference workers. Parallelism is fixed to
1; external hosts and speculative-decoding options are rejected.

The shipped LRFU profile uses a 12-slot decode window, prediction depth 3 and
24 protected slots per layer. Positive protected slots require that combination
and enabled prefetch. The policy scores actual router demand with a 32-step
half-life; it does not change routing top-k, expert weights or precision.

| Profile setting | Default and purpose |
|---|---|
| `host`, `port` | `127.0.0.1`, `8080`; localhost only |
| `args.model`, `args.moe-experts` | First GGUF shard and matching expert directory/manifest |
| `args.alias` | `moeshift`; must equal its key under `models` |
| `args.ctx-size`, `args.ctx-checkpoints` | `65536` tokens; `1` checkpoint |
| `args.threads` | `10` CPU threads |
| `args.prefill-max-tokens` | `3072` tokens per prefill microbatch |
| `args.moe-prefill-bank`, `args.moe-decode-bank` | `512` shared expert slots during prefill; `0` shared slots retained during decode |
| `args.n-gpu-layers`, `args.flash-attn` | `99`, `on`; expert execution requires Metal |
| `args.cache-type-k`, `args.cache-type-v` | `q8_0` KV quantization; the indexer stores keys only |
| `args.no-mmap` | `true`; explicit reads for Qwen weights |
| `env.LLAMA_MOE_DECODE_WINDOW` | `12` decode slots per layer |
| `env.LLAMA_MOE_PROTECTED_SLOTS` | `24` additional protected slots; `0` disables the protected tier |
| `env.LLAMA_MOE_LOOKAHEAD_PREFETCH` | `1`; enable predictive expert reads |
| `env.LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH`, `env.LLAMA_MOE_PREFETCH_READ` | Predict `3` experts, read `2` |
| `env.LLAMA_MOE_PREFILL_PIPELINE` | `1`; overlap prefill reads with computation |
| `env.LLAMA_GPU_KEEPALIVE` | `1`; keep the GPU active while the worker exists; `0` disables it |
| `env.LLAMA_MOE_QUIET` | `1`; suppress expert diagnostics; `0` enables them |

Use the argument names as CLI overrides (`--ctx-size`, `--threads`, etc.).
`--no-mmap true/false` overrides its JSON boolean. Cache overrides are
`--moe-decode-window` and `--moe-protected-slots`. Other environment entries are
set in the profile. Stop with Ctrl-C. The worker, its readers and the GPU
keepalive terminate with the server, including after a forced server exit.

## HTTP

```sh
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"moeshift","messages":[{"role":"user","content":"Ciao"}],"max_tokens":128,"stream":true}'
```

Use the profile identifier returned by `/v1/models`. Streaming uses SSE and ends
with `data: [DONE]`; `stream: false` returns a single JSON response.
`/v1/completions` accepts a `prompt` instead of `messages`.

## Upstream and license

MoEShift builds on [tc-mb/llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni),
[llama.cpp](https://github.com/ggml-org/llama.cpp) and
[ggml](https://github.com/ggml-org/ggml). The inherited MoE-SSD/LES foundations
precede MoEShift's Qwen runtime, Metal scheduling and cache changes. The protected
cache adapts the demand-weighted decay idea from
[LRFU](https://www.openu.ac.il/home/wiseman/2os/lru/lrfu.pdf).

The original Git history and copyright notices are retained. See [LICENSE](LICENSE)
and the bundled third-party licenses. Model weights retain their own licenses.
