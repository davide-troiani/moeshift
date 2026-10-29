#!/usr/bin/env python3
"""Integration check with real weights: a worker must exit if its server is killed."""
import http.client
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time

binary, config = (str(Path(arg).resolve()) for arg in sys.argv[1:])
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
with tempfile.TemporaryFile(mode="w+") as log:
    parent = subprocess.Popen([binary, "--config", config, "--port", str(port)],
                              stdout=log, stderr=log, cwd="/private/tmp")
    worker = None
    try:
        for _ in range(100):
            try:
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
                conn.request("GET", "/health")
                ready = conn.getresponse().status == 200
                conn.close()
                if ready:
                    break
            except OSError:
                pass
            time.sleep(0.05)
        else:
            raise AssertionError("Server did not start")
        model = next(iter(json.loads(Path(config).read_text())["models"]))
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
        conn.request("POST", "/v1/chat/completions", json.dumps({
            "model": model, "messages": [{"role": "user", "content": "Say hi."}],
            "max_tokens": 4, "chat_template_kwargs": {"enable_thinking": False}}),
            {"Content-Type": "application/json"})
        response = conn.getresponse()
        body = response.read()
        conn.close()
        log.seek(0)
        output = log.read()
        match = re.search(r"loading .* \(pid (\d+)\)", output)
        assert match, output
        worker = int(match.group(1))
        assert response.status == 200, body
        assert "keepalive: ready" in output, "Test requires an active GPU keepalive"
        parent.kill()
        parent.wait(timeout=5)
        for _ in range(60):
            try:
                os.kill(worker, 0)
            except ProcessLookupError:
                print("PASS: worker and GPU keepalive exited after server SIGKILL")
                break
            time.sleep(0.05)
        else:
            raise AssertionError("Orphan worker kept running after server SIGKILL")
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        if worker is not None:
            try:
                os.kill(worker, signal.SIGKILL)
            except ProcessLookupError:
                pass
