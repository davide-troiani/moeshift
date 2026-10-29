# MoEShift configuration

The default model identifier and alias are `qwen-3.8-flash-next-rco-gsq-iq3-xxs`.
Run `moeshift` to use `~/.config/moeshift/config.json` by default.
If `XDG_CONFIG_HOME` is set, the user path becomes `$XDG_CONFIG_HOME/moeshift/config.json`.
Set `args.model` to the first GGUF shard and `args.moe-experts` to the prepared folder.

Lookup order: explicit `--config FILE`, `moeshift.json` in the current directory,
`$XDG_CONFIG_HOME/moeshift/config.json`
(or `~/.config/moeshift/config.json`), then the installed default.
Profile paths resolve relative to the configuration file.

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
| `args.model`, `args.moe-experts` | First GGUF shard and matching expert directory/manifest; required paths |
| `args.alias` | `qwen-3.8-flash-next-rco-gsq-iq3-xxs`; must equal its key under `models` |
| `args.ctx-size`, `args.ctx-checkpoints` | `65536` tokens; `1` checkpoint |
| `args.threads` | `10` CPU threads |
| `args.prefill-max-tokens` | `3072` tokens per prefill microbatch |
| `args.moe-prefill-bank`, `args.moe-decode-bank` | `512` shared expert slots during prefill; larger banks use more RAM; `0` shared slots retained during decode |
| `args.n-gpu-layers`, `args.flash-attn` | `99`, `on`; expert execution requires Metal |
| `args.cache-type-k`, `args.cache-type-v` | `q8_0` KV quantization; the indexer stores keys only |
| `args.no-mmap` | `true`; explicit reads for Qwen weights |
| `env.LLAMA_MOE_DECODE_WINDOW` | `12` decode slots per layer; combined with 3 prefetch slots and 24 protected slots, the default total is 39 |
| `env.LLAMA_MOE_PROTECTED_SLOTS` | `24` additional protected slots; larger values use more RAM and can reduce SSD reads; `0` disables the protected tier |
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

The `env` entries above belong inside the JSON profile, not in your shell environment. The installed template is `share/moeshift/moeshift.json`. On first launch, if no current-directory or user configuration exists, MoEShift copies that template to the user path without overwriting an existing file. Edit its model paths before inference.
