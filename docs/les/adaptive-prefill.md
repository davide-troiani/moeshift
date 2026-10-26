# Adaptive LES prefill

`llama-les-server` supports adaptive prefill for Qwen4Exp on macOS/Metal,
with one inference slot and speculative decoding disabled.

| Startup option | Default | Behavior |
|---|---:|---|
| `--prefill-max-tokens` | 3072 | Maximum logical batch and physical microbatch size. Overrides `--batch-size` and `--ubatch-size`. |
| `--moe-prefill-bank` | 512 | Shared prefill expert slots. Overrides `--moe-slot-bank`. |
| `--moe-decode-bank` | 0 | Shared bank slots retained during decoding; must be smaller than the prefill bank. |

The server fills batches with the available prompt tokens, up to the configured
limit. Existing context checkpoints may split them further. A shared bank of
512 slots covers all experts in the tested Qwen model.

Keep the baseline environment:

```sh
export LLAMA_GPU_KEEPALIVE=1
export LLAMA_MOE_DECODE_WINDOW=10
export LLAMA_MOE_F_NOCACHE=0
export LLAMA_MOE_LOOKAHEAD_PREFETCH=1
export LLAMA_MOE_PREFETCH_READ=2
export LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH=3
export LLAMA_MOE_PREFILL_PIPELINE=1
```

Use `--parallel 1 --spec-type none`, and pass the existing model, experts,
context and cache arguments to `llama-les-server`. The three adaptive options
above are optional because their defaults enable the tested configuration.

After prefill, the server synchronizes GPU work and expert readers, invalidates
cached mappings, and releases the unused shared-bank Metal ranges. Decode bank
zero retains the separate per-layer decode windows: 10 cache slots plus 3
prefetch slots in this configuration. Virtual addresses remain reserved; excluded
host pages become reusable by macOS. The next prefill restores the full bank.

`LES adaptive-batch`, `adaptive phase`, and `LES phase-memory` log the actual
microbatches, shared bank size, and active/released allocation bytes.

Validation on Apple M4 Pro, 24 GiB, with context capacity 65536 and 128 generated
tokens per request, without prompt-cache reuse or an A/B/A comparison:

| Prompt | Tokens | Actual prefill batches | Prefill token/s | Decode token/s | Peak decode footprint, decimal GB |
|---|---:|---|---:|---:|---:|
| Long text | 5929 | 2857, 3068, 4 | 191.512 | 7.810 | 7.473 |
| Short sentence | 12 | 8, 4 | 8.105 | 8.163 | 6.503 |

Both requests reduced the shared bank from 512 to zero, releasing 1,192,722,432
bytes of Metal allocations while preserving the decode windows and prefetch.
The short request followed the long one in the same process, exercising bank
restoration. These measurements do not establish a speedup against a control.
Other models and positive decode-bank values were not benchmarked.

The macOS test `test-les-active-ranges` checks shrink/restore, stable tensor
addresses, and invalid-range rejection without loading a model:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON
cmake --build build --target llama-les-server test-les-active-ranges -j 8
ctest --test-dir build -R '^test-les-active-ranges$' --output-on-failure
```
