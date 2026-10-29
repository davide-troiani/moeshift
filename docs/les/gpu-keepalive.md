# GPU keepalive

During the short I/O waits between expert reads, the GPU can lower its clock
frequency. On the tested **M4 Pro**, it did not return to maximum frequency
quickly enough when computation resumed. Keeping the GPU active avoids those
repeated frequency drops and helps sustain maximum frequency during inference.
This behavior has been tested only on M4 Pro.

## Configuration

The model profile enables keepalive by default:

```json
"env": {
  "LLAMA_GPU_KEEPALIVE": 1
}
```

Set this entry to `0` to disable it. Edit the JSON configuration; shell exports
are not inherited by the inference worker. Keepalive increases idle GPU activity
and power consumption while the model worker exists.

## Implementation and lifetime

`tools/server/gpu-keepalive.mm` runs a small compute kernel on a separate Metal
queue and a 512-byte buffer. It does not access model tensors. Each command uses
one 32-thread threadgroup and 500,000 FMA iterations; the CPU waits for completion.

Keepalive starts before model loading. Initialization failure prevents worker
startup; subsequent Metal errors are logged and end the keepalive thread.
Normal worker shutdown signals and joins the thread, including on model-load
failure. The parent-process guard terminates the inference worker if the main
server disappears, so its readers and keepalive do not outlive the server.

The process-lifetime check is `tests/test-moeshift-parent.py`; it requires a real
model configuration and tests parent termination with keepalive enabled.
