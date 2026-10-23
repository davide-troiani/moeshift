# LES server GPU keepalive

Apple builds with `GGML_METAL` enable a continuous GPU keepalive in
`llama-les-server`. Set `LLAMA_GPU_KEEPALIVE=0` to disable it. Other targets
and non-Metal builds are unchanged.

The worker uses its own Metal queue and a 512-byte buffer, with one threadgroup
of 32 threads executing 500000 FMA iterations per command. It never accesses
model tensors and waits for each command without spinning on the CPU.

It starts before model loading and joins on server shutdown, including model
loading failures. Initialization failure prevents server startup; later Metal
errors are logged and stop the worker. The keepalive stays active while the
server is idle and increases GPU power consumption. Do not run the standalone
keepalive helper alongside this integrated worker.

The isolated prototype produced bit-identical logits for 256 generated tokens
against the native baseline, with a measured peak footprint of 8.64 GB on the
Mac mini M4 Pro test configuration. This is a workload-specific measurement,
not a memory limit or a guarantee of decoding speed on other configurations.
