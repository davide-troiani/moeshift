# Protected decode cache

The [supported SSD architectures](../architecture.md#supported-llm-architectures)
use the same two-tier, demand-scored LRFU cache.
Configure it inside the model configuration's `env` object in
`~/.config/moeshift/config.json`:

```json
{
  "LLAMA_MOE_DECODE_WINDOW": 12,
  "LLAMA_MOE_LOOKAHEAD_PREFETCH": 1,
  "LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH": 3,
  "LLAMA_MOE_PROTECTED_SLOTS": 24
}
```

These are model configuration entries, not shell exports. The router isolates workers from
inherited tuning variables. Positive protected slots require window 12,
prefetch enabled and depth 3. Set protected slots to `0` to disable the
protected tier. The CLI override is `--moe-protected-slots N`.

## Slot ownership

Each enabled layer or loop visit budgets **15 recent slots** (12 decode + 3
prefetch). The Qwen presets add **24 protected slots** and the Inkling preset
adds **6 protected slots**. Recent and protected are logical groups. Promotion
and demotion change metadata without moving expert weights or tensor addresses.

- A recent hit updates recency.
- A protected hit promotes the expert to recent, demoting the oldest eligible
  recent expert when necessary.
- A miss uses a free recent slot if possible. Otherwise it selects an eligible
  recent victim and considers demoting it into the protected group.
- With no free physical slot, protected admission compares that recent expert's
  score with the lowest-scoring eligible protected expert. The recent expert is
  retained if its score is at least as high; otherwise the recent slot is reused.
- Protected score ties use age. Victim selection excludes requested experts and
  pending reads, and prefers avoiding predicted experts.

Prefetch uses the same slot policy. Pending reads pin their slots until
publication. Phase changes synchronize readers and GPU work before invalidating
mappings and score history.

With the Inkling 2 Mini preset, 48 visits use 24 decode windows. Each block
inside the loop uses a single window with 45 recent and 18 protected slots.
Other blocks use 21 slots each. The
pooled window combines the visits' budgets, capped at the smaller of the prefill
bank and expert count. An expert loaded by one visit is immediately available
to the next visit to that block. The asynchronous reader fetches missing
experts from SSD and prefetch avoids evicting slots used by the active visit.
Each visit keeps its own attention and convolution state.

## Demand LRFU scoring

Before slot assignment, each layer or loop visit updates one score per expert:

```text
score[expert] = 2^(-1/32) * score[expert] + is_requested[expert]
```

Every expert selected by the actual router receives one vote, whether hit or
miss. Prefetch alone adds no votes. The fixed half-life is 32 updates, or 32
visits to a shared loop window. A vote retains 1/16 of its weight after 128 updates.
This adapts the recency-weighted
frequency idea from [LRFU](https://www.openu.ac.il/home/wiseman/2os/lru/lrfu.pdf)
to protected admission and eviction. It does not change expert selection,
weights or precision. The half-life is not a measured optimum for every workload.

## Capacity and memory

The protected-slot limit is `min(prefill bank, expert count, 1024) - 15`.
With the example prefill banks, valid values are 0–497 for Flash Next,
0–241 for Qwen3.6 and 0–113 for Inkling 2 Mini. Invalid capacities are rejected.

| Tested quantization | Expert layers | Extra memory per protected slot across layers |
|---|---:|---:|
| Flash Next GSQ-RCO IQ3_XXS | 48 | 83.8144 MB |
| Qwen3.6 UD Q6_K XL | 40 | 113.8852 MB |
| Inkling 2 Mini UD-Q4_K_XL | 48 visits over 24 blocks | 230.9161 MB |

These are expert tensor bytes (decimal). On Inkling, the shared window cap limits
further allocation in the looped blocks as the configured capacity increases.

Set `env.LLAMA_MOE_QUIET` to `0` to see `protected-cache` diagnostics.
The `loop-cache` line gives the number of Inkling decode windows and visits.

## Code and verification

- Policy: `src/llama-moe-protected-cache.h`.
- Demand, prefetch and phase integration: `src/llama-moe-slot.cpp`.
- Policy checks: `tests/test-les-protected-cache.cpp`.

```sh
cmake --build build --target test-les-protected-cache -j 8
ctest --test-dir build -R '^test-les-protected-cache$' --output-on-failure
```
