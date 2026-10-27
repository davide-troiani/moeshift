# Protected decode cache

Adaptive Qwen4Exp LES enables a two-tier cache when the per-layer decode window
is 12, prefetch depth is 3, and unified prefetch is enabled. Other window sizes
and partitioned prefetch retain the existing cache policy.

Each enabled layer has 15 recent slots and a configurable number of protected
slots. Set `LLAMA_MOE_PROTECTED_SLOTS` to a nonnegative integer; the default is
23. Zero disables the protected tier and restores the existing unified LRU.
The maximum is `min(prefill bank, number of experts, 1024) - 15`: with this
model and bank 512 the allowed range is 0–497. Invalid values are rejected at
model loading. The setting applies only to layers with the 12+3 configuration;
other layers retain their existing policy.

```sh
LLAMA_MOE_PROTECTED_SLOTS=32  # 32 protected slots per eligible layer
```

For the Qwen Flash IQ3 expert files used in the baseline, one slot holds one
expert's gate, up and down tensors. Their quantization varies by layer:

| MB per slot in one layer | Number of layers |
|---:|---:|
| 1.3056 | 8 |
| 1.4080 | 6 |
| 1.5104 | 6 |
| 1.7152 | 3 |
| 1.7664 | 1 |
| 1.8688 | 11 |
| 1.9712 | 4 |
| 2.1760 | 3 |
| 2.3296 | 6 |

An extra slot in **each of the 48 layers** reserves 83.8144 MB (decimal).
Thus 23 slots per layer reserve 1,927.7312 MB and 32 reserve 2,682.0608 MB,
plus small metadata. These are expert tensor allocation sizes, not a guarantee
of the process's physical footprint; model, quantization, alignment, other
allocations and OS accounting affect the actual result. The decode phase log
reports the configured slot count and the total expert bytes for active layers.

## Cache assignment: before and after

Before the protected tier, demand and prefetch shared one per-layer LRU window:

```mermaid
flowchart TD
    A[Expert requested] --> B{Already resident?}
    B -->|Yes| C[Use slot and update recency]
    B -->|No| D{Free slot?}
    D -->|Yes| E[Read expert into free slot]
    D -->|No| F[Evict least-recent eligible expert]
    F --> E
    E --> C
    F -.-> X[Skip requested experts and pending reads; prefer avoiding predicted experts]
```

The protected tier uses two **logical groups in the same physical slots**.
At each decode step, it updates the exponentially aged demand scores before
allocating slots; prefetch does not update those scores.

```mermaid
flowchart TD
    A[Expert requested] --> B{Resident group?}
    B -->|Recent| C[Use slot; update recency]
    B -->|Protected| E{Recent slot can be demoted?}
    E -->|Yes| F[Promote expert and demote oldest eligible recent expert]
    E -->|No| C
    F --> C
    B -->|Absent| H{Recent group below 15 and free slot?}
    H -->|Yes| I[Read into free recent slot]
    H -->|No| J[Select oldest eligible recent expert]
    J --> K{Free physical slot?}
    K -->|Yes| L[Demote recent expert to protected; read into free slot]
    K -->|No| M[Select protected expert with lowest frequency; oldest breaks ties]
    M --> N{Recent frequency at least protected frequency?}
    N -->|Yes| O[Demote recent expert; replace protected expert]
    N -->|No or none eligible| P[Replace recent expert]
    O --> Q[Read into released slot]
    P --> Q
    I --> C
    L --> C
    Q --> C
    J -.-> R[Skip requested experts and pending reads; prefer avoiding predicted experts]
    M -.-> R
```

Prefetch uses the same slot policy but does not know the target layer's current
request set when it claims a slot. A pending slot is unavailable for eviction;
a completed prefetch publishes its mapping. Promotions and demotions change
metadata rather than moving weights. Prefill/decode phase changes synchronize
readers and GPU work before invalidating mappings and frequency history.

## Demand LRFU

Protected admission and eviction use exponentially aged **actual router demand**.
Slot counts and the enabling conditions above do not change.

For each layer and decode token, before assigning any slots:

```text
score[expert] = 2^(-1/32) * score[expert] + is_requested_this_token[expert]
```

All demanded experts receive exactly one vote, including recent hits, protected
hits and misses. Prefetch claims/publications receive no votes. The existing
15-slot recent group, protected victim search, admission comparison, request
pins, prediction preference and pending-read ownership remain in place. Thus
this is an LRFU scoring/admission adaptation inside the existing two-tier cache,
not a replacement of the whole cache with textbook LRFU.

The score includes repeated recent hits and excludes unconsumed speculative reads.
A historical vote retains 1/16 of its weight after 128 tokens. The 32-token
half-life is an initial choice, **not a measured optimum for Qwen**;
short memory can discard useful experts with long reuse intervals.

The [LRFU paper](https://www.openu.ac.il/home/wiseman/2os/lru/lrfu.pdf) supplies
the recency-weighted frequency idea. The model-specific adaptation here is to
count a complete Qwen4Exp top-10 router set on the per-layer token clock, using
an exact 512-entry score vector rather than an approximate sketch. No routing,
expert weights, precision or accumulation order changes.

### Memory and trying the prototype

The score vector already exists: 512 floats × 48 layers = 98,304 bytes for
the protected cache. No new expert slots, tensor buffers or history arrays are allocated.
At 32 protected slots, the protected tensors remain 2,682,060,800 bytes; at the
unchanged default of 23 they remain 1,927,731,200 bytes. **Equal allocations do
not prove a process footprint below 10,000,000,000 bytes.**