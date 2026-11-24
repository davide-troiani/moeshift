# Adaptive prefill

MoEShift uses different expert-memory budgets for reading a prompt (prefill)
and generating a reply (decode). The [supported SSD architectures](../architecture.md#supported-llm-architectures)
use this path on Apple Silicon with Metal. Inference parallelism is fixed to one.

## Configuration

Edit the model configuration in `~/.config/moeshift/config.json`:

| Setting | Default | Effect |
|---|---:|---|
| `args.prefill-max-tokens` | `3072` | Maximum logical batch and physical microbatch size. Smaller batches reduce peak working memory. |
| `args.moe-prefill-bank` | `512` / `256` | Shared expert slots used during prefill. A larger bank can reduce SSD reads, using more memory. |
| `args.moe-decode-bank` | `0` | Shared expert slots retained during decode; must be smaller than the prefill bank. Decode windows have a separate allocation. |
| `env.LLAMA_MOE_PREFILL_PIPELINE` | `1` | Overlap prefill expert reads with computation. |

The three `args` entries also accept CLI overrides with the corresponding
`--` names. Runtime tuning is read from JSON, not inherited shell exports.
The SSD presets use window 12 and prefetch depth 3, with 24 protected slots
for Qwen and 6 for Inkling. See [protected decode cache](protected-decode-cache.md).

## Execution and phase transitions

The server fills each batch with available prompt tokens up to the configured
limit. When checkpoints are enabled, Inkling adds checkpoint boundaries to
requests that use prompt caching. The shared prefill bank covers 512 experts
for Flash Next, 256 for Qwen3.6, and 128 for Inkling 2 Mini.

Inkling groups adjacent experts from the packed file into a single vectored
read, writing each one straight into its slot. With the prefill pipeline
enabled, batches of at least 512 tokens start staging a full layer when at
least 75% of its experts are needed. A dense layer also starts the next read
while Metal computes. Sparse routing uses the bounded reads when there is no
staged read for the current layer. The Qwen profiles use full-layer reads
starting at 1,024 tokens.

The tiled Metal Flash Attention kernel looks up Inkling's relative bias from
the projected table as it consumes each causal or sliding-window mask tile.
The [Inkling preset](../../configs/inkling-2-mini.json) uses a 3,072-token batch
limit to reuse each expert read across prompt tokens.

Before switching phases, the runtime synchronizes GPU work and drains expert
readers. It invalidates cached mappings and sets the active Metal buffer ranges
for the new phase. During decode, the unused shared-bank ranges are released;
the decode windows and protected cache stay active. Virtual
addresses stay reserved while discarded host pages become reusable by macOS.
The next prefill restores the shared-bank ranges before using them.

Expert execution uses the common Metal phase identifiers in
`ggml/src/ggml-metal/ggml-metal-impl.h`: `ALL` (0), `RESIDENT` (1), and
`MISSING` (2). The resident/missing path requires Metal placement. Predictions
schedule reads; the actual router determines which experts are computed.

In Inkling's single-token SSD decode, Metal fuses the L2 clipping of the
attention queries, keys and relative-position queries. It also fuses SiTU
within each expert phase, so the existing readiness dependency still gates
the missing experts. Unsupported tensor layouts use the ordinary operations.

Set `env.LLAMA_MOE_QUIET` to `0` for diagnostics such as `adaptive-batch`,
`adaptive phase=prefill/decode`, `phase-scratch`, and `protected-cache`.

The pipeline stages at most 1,200 MiB of expert data per layer; larger layers
use bounded slot reads instead.

## Code and verification

The server sets batch and bank budgets in `tools/server/server-les.cpp`.
`src/llama-moe-slot.cpp` handles phase transitions and expert readers;
`src/llama-context.cpp` coordinates the graph and phase scratch memory.

With tests enabled in the build:

```sh
cmake --build build --target moeshift test-les-active-ranges test-metal-moe-phases -j 8
ctest --test-dir build -R '^(test-les-active-ranges|test-metal-moe-phases)$' --output-on-failure
```

These checks cover Metal active-range shrink/restore and shared phase constants.
They are not end-to-end performance benchmarks; see the repository README for
measured inference results.
