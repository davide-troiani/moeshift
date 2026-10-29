# Adaptive prefill

MoEShift uses different expert-memory budgets for reading a prompt (prefill)
and generating a reply (decode). This path targets Qwen 3.8 Flash Next on
Apple Silicon with Metal. Inference parallelism is fixed to one.

## Configuration

Edit the model profile in `~/.config/moeshift/config.json`:

| Profile entry | Default | Effect |
|---|---:|---|
| `args.prefill-max-tokens` | `3072` | Maximum logical batch and physical microbatch size. Smaller batches reduce peak working memory. |
| `args.moe-prefill-bank` | `512` | Shared expert slots used during prefill. A larger bank can reduce SSD reads, using more memory. |
| `args.moe-decode-bank` | `0` | Shared expert slots retained during decode; must be smaller than the prefill bank. Per-layer decode windows remain allocated separately. |
| `env.LLAMA_MOE_PREFILL_PIPELINE` | `1` | Overlap prefill expert reads with computation. |

The three `args` entries also accept CLI overrides with the corresponding
`--` names. Runtime tuning is read from JSON, not inherited shell exports.
The shipped decode profile uses window 12, prefetch depth 3 and 24 protected
slots; see [protected decode cache](protected-decode-cache.md).

## Execution and phase transitions

The server fills each batch with available prompt tokens up to the configured
limit. Context checkpoints can split it further. A 512-slot shared bank covers
the reference model's 512 experts and is reused across layers.

Before switching phases, the runtime synchronizes GPU work and drains expert
readers. It invalidates cached mappings and sets the active Metal buffer ranges
for the new phase. During decode, the unused shared-bank ranges are released;
the per-layer decode windows and protected cache remain available. Virtual
addresses stay reserved while discarded host pages become reusable by macOS.
The next prefill restores the shared-bank ranges before using them.

Expert execution uses the common Metal phase identifiers in
`ggml/src/ggml-metal/ggml-metal-impl.h`: `ALL` (0), `RESIDENT` (1), and
`MISSING` (2). The resident/missing path requires Metal placement. Predictions
schedule reads; the actual router determines which experts are computed.

Set `env.LLAMA_MOE_QUIET` to `0` for diagnostics such as `adaptive-batch`,
`adaptive phase=prefill/decode`, `phase-scratch`, and `protected-cache`.

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
