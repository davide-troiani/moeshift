# Protected decode cache

Adaptive Qwen4Exp LES enables a two-tier cache when the per-layer decode window
is 12, prefetch depth is 3, and unified prefetch is enabled. Other window sizes
and partitioned prefetch retain the existing cache policy.

Each enabled layer has 38 GPU-visible expert slots: 15 recent and 23 protected.
The protected tier keeps experts requested repeatedly after leaving the recent
tier. Frequency counts halve every 128 decode steps. Promotions change metadata,
not weight addresses; no additional weight copies are needed. Replacement skips
experts required by the current batch and slots owned by pending reads.

The protected capacity is fixed at 23 slots per layer; there is no CLI or
environment parameter to resize it. `LLAMA_MOE_DECODE_WINDOW=12` remains unchanged.
For the tested Qwen model this adds 1,927,731,200 bytes of expert capacity, plus
small metadata. Prefill and decode activate their separate Metal memory ranges.
Cache mappings and frequency history reset at phase transitions.

One candidate-only measurement on Apple M4 Pro (24 GiB), with keepalive enabled,
context capacity 65536, and a short Italian prompt about virtual memory:

| Metric | Result |
|---|---:|
| Prompt including chat template | 77 tokens |
| Generated, including reasoning | 300 tokens |
| Decode | 9.0767 tokens/s |
| Decode time | 33.052 s |
| Peak decode physical footprint | 9.5835 decimal GB |

This is not a baseline comparison or proof of improved throughput or long-run
stability. Full-logit equivalence was not checked. OS file cache was not cleared;
preliminary server starts and an interrupted request preceded the completed run.

The standalone `test-les-protected-cache` checks capacity, pinned requests,
pending-read ownership, map consistency, reset, and frequency decay without a
model or GPU.
