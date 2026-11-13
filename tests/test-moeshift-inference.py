#!/usr/bin/env python3
"""Offline synthetic inference checks; requires the existing gguf-py dependencies.

Usage: python test-moeshift-inference.py BUILD_BIN [BASELINE_COMPLETION]
Synthetic weights exercise execution, not model quality or performance.
"""
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gguf-py"))
from gguf import GGUFWriter


def model(path, arch):
    rng = np.random.default_rng(17)
    writer = GGUFWriter(path, arch)
    tokens = ["<unk>", "<s>", "</s>"] + [f"<0x{i:02X}>" for i in range(256)] + ["▁"] + list("abcdefghijklmnopqrstuvwxyz")
    writer.add_tokenizer_model("llama")
    writer.add_token_list(tokens)
    writer.add_token_types([2, 3, 3] + [6] * 256 + [1] * 27)
    writer.add_token_scores([0.0] * len(tokens))
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_unk_token_id(0)
    writer.add_add_bos_token(True)
    writer.add_chat_template("{{ bos_token }}{% for m in messages %}{{ m['content'] }}{% endfor %}")
    writer.add_context_length(256)
    writer.add_embedding_length(32)
    writer.add_block_count(1)
    writer.add_head_count(4 if arch != "mamba" else 0)
    writer.add_layer_norm_rms_eps(1e-5)

    def tensor(name, shape, norm=False):
        data = np.ones(shape, dtype=np.float32) if norm else (rng.standard_normal(shape) * .05).astype(np.float32)
        if name == "output.weight":
            data[:259] = 0  # Keep synthetic output in printable token pieces.
        writer.add_tensor(name, data)

    tensor("token_embd.weight", (len(tokens), 32))
    tensor("output.weight", (len(tokens), 32))
    tensor("output_norm.weight", (32,), True)
    tensor("blk.0.attn_norm.weight", (32,), True)
    if arch == "mamba":
        writer.add_ssm_conv_kernel(4)
        writer.add_ssm_inner_size(64)
        writer.add_ssm_state_size(16)
        writer.add_ssm_time_step_rank(2)
        for name, shape in {
            "ssm_in.weight": (128, 32), "ssm_conv1d.weight": (64, 4),
            "ssm_conv1d.bias": (64,), "ssm_x.weight": (34, 64),
            "ssm_dt.weight": (64, 2), "ssm_dt.bias": (64,),
            "ssm_a": (64, 16), "ssm_d": (64,), "ssm_out.weight": (32, 64),
        }.items():
            tensor("blk.0." + name, shape)
    else:
        writer.add_feed_forward_length(64)
        writer.add_head_count_kv(4)
        writer.add_rope_dimension_count(8)
        for name in ("attn_q", "attn_k", "attn_v", "attn_output"):
            tensor("blk.0." + name + ".weight", (32, 32))
        tensor("blk.0.ffn_norm.weight", (32,), True)
        if arch == "qwen3moe":
            writer.add_expert_count(4)
            writer.add_expert_used_count(2)
            writer.add_expert_feed_forward_length(64)
            for name in ("attn_q_norm", "attn_k_norm"):
                tensor("blk.0." + name + ".weight", (8,), True)
            tensor("blk.0.ffn_gate_inp.weight", (4, 32))
            for name, shape in (("gate", (4, 64, 32)), ("up", (4, 64, 32)), ("down", (4, 32, 64))):
                tensor("blk.0.ffn_" + name + "_exps.weight", shape)
        else:
            for name, shape in (("gate", (64, 32)), ("up", (64, 32)), ("down", (32, 64))):
                tensor("blk.0.ffn_" + name + ".weight", shape)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def request(port, method, path, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        conn.request(method, path, None if body is None else json.dumps(body), {"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read()
        assert response.status == 200, data
        return json.loads(data)
    finally:
        conn.close()


if __name__ == "__main__":
    binaries = Path(sys.argv[1]).resolve()
    baseline = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None
    with tempfile.TemporaryDirectory(prefix="moeshift-inference-") as tmp:
        root = Path(tmp)
        profiles = {}
        for arch in ("llama", "qwen3moe", "mamba"):
            path = root / (arch + ".gguf")
            model(path, arch)
            args = {"model": str(path), "alias": arch, "ctx-size": 128, "n-gpu-layers": 0, "threads": 2}
            profiles[arch] = {"args": args}
            profiles[arch + "-offload"] = {"args": {**args, "alias": arch + "-offload", "n-gpu-layers": 99}}
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        config = root / "config.json"
        config.write_text(json.dumps({"host": "127.0.0.1", "port": port, "models": profiles}))
        env = {k: v for k, v in os.environ.items() if not k.startswith(("LLAMA_", "GGML_", "LES_", "MB_"))}
        with (root / "server.log").open("w+", errors="replace") as log:
            proc = subprocess.Popen([str(binaries / "moeshift"), "--config", str(config)], env=env, stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 20
                while True:
                    try:
                        request(port, "GET", "/health")
                        break
                    except OSError:
                        assert proc.poll() is None and time.monotonic() < deadline, "Server did not start"
                        time.sleep(.05)
                for arch in profiles:
                    body = {"model": arch, "prompt": "abc", "max_tokens": 8, "temperature": 0, "seed": 1}
                    result = request(port, "POST", "/v1/completions", body)
                    assert result["usage"]["completion_tokens"] > 0
                    if baseline:
                        command = [str(baseline), "-m", profiles[arch]["args"]["model"], "-p", "abc",
                                   "-n", "8", "-c", "128", "-ngl", "0", "-t", "2", "--temp", "0", "--seed", "1",
                                   "--no-display-prompt", "--no-warmup", "-no-cnv"]
                        expected = subprocess.run(command, env=env, capture_output=True, timeout=30, check=True)
                        assert expected.stdout.decode().strip() == result["choices"][0]["text"].strip(), (arch, expected.stdout, result)
                    chat = request(port, "POST", "/v1/chat/completions", {
                        "model": arch, "messages": [{"role": "user", "content": "abc"}], "max_tokens": 4, "temperature": 0})
                    assert chat["usage"]["completion_tokens"] > 0
                    print("PASS synthetic prefill/decode, chat and completions:", arch)
            except BaseException:
                log.seek(0)
                print(log.read(), file=sys.stderr)
                raise
            finally:
                proc.terminate()
                proc.wait(timeout=15)
