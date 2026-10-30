#!/usr/bin/env python3
"""Benchmark runner checks with a local fake MoEShift process; no model needed."""

import http.server
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'benchmarks/run.py'


def fake_server():
    args = sys.argv
    config = json.loads(Path(args[args.index('--config') + 1]).read_text())
    port = int(args[args.index('--port') + 1])
    calls = 0

    def event(value):
        with open(config['events'], 'a') as output:
            output.write(json.dumps(value) + '\n')

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        def do_POST(self):
            nonlocal calls
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls += 1
            event({'request': body, 'call': calls})
            if config.get('mode') == 'slow':
                time.sleep(60)
            if config.get('mode') == 'http-error':
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b'/Users/private-user/secret-model')
                return
            timings = dict(cache_n=0, prompt_n=60, prompt_ms=100 * calls,
                           prompt_per_second=600 / calls, predicted_n=512,
                           predicted_ms=1000 * calls, predicted_per_second=512 / calls)
            if config.get('mode') == 'early-stop':
                timings['predicted_n'] = 64
            if config.get('mode') == 'cached':
                timings['cache_n'] = 5
            if config.get('mode') == 'invalid':
                timings['predicted_per_second'] = None
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({
                'timings': timings, 'usage': {'prompt_tokens': 60, 'completion_tokens': timings['predicted_n']},
                'choices': [{'finish_reason': 'stop' if config.get('mode') == 'early-stop' else 'length', 'message': {'content': 'private-user'}}],
                'system_fingerprint': '/Users/private-user/build',
            }).encode())

    def stop(*_):
        event({'stopped': os.getpid()})
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    event({'started': os.getpid()})
    if config.get('mode') == 'exit':
        return
    with http.server.HTTPServer(('127.0.0.1', port), Handler) as server:
        server.serve_forever(poll_interval=0.05)


class BenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='benchmark test ')
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.events = self.directory / 'events.jsonl'
        self.config = self.directory / 'config.json'
        self.config.write_text(json.dumps({'events': str(self.events)}))
        self.binary = self.directory / 'fake moeshift'
        self.binary.write_text(
            f'#!{sys.executable}\nimport runpy, sys\nsys.argv.insert(1, "--fake-server")\n'
            f'runpy.run_path({str(Path(__file__).resolve())!r}, run_name="__main__")\n')
        self.binary.chmod(0o755)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            self.port = sock.getsockname()[1]
        self.output = self.directory / 'report.json'

    def command(self):
        return [sys.executable, str(RUNNER), '--binary', str(self.binary),
                '--config', str(self.config), '--port', str(self.port), '--output', str(self.output)]

    def run_benchmark(self):
        return subprocess.run(self.command(), capture_output=True, text=True, timeout=20)

    def read_events(self):
        return [json.loads(line) for line in self.events.read_text().splitlines()]

    def test_sequence_medians_and_privacy(self):
        result = self.run_benchmark()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(self.output.read_text())
        self.assertEqual(set(report), {'schema_version', 'protocol', 'benchmarks'})
        self.assertNotIn('private-user', self.output.read_text())
        self.assertNotIn(str(self.directory), self.output.read_text())
        self.assertEqual([item['name'] for item in report['benchmarks']], ['short', 'long'])
        for item in report['benchmarks']:
            self.assertEqual(len(item['runs']), 3)
            self.assertEqual(item['warmup']['timings']['prompt_ms'], 100)
            self.assertEqual(item['median']['prompt_ms'], 300)
            self.assertEqual(item['median']['predicted_ms'], 3000)
            self.assertEqual(item['median']['inference_ms'], 3300)
            self.assertEqual(len(item['payload_sha256']), 64)
        events = self.read_events()
        self.assertEqual(len([e for e in events if 'started' in e]), 2)
        self.assertEqual(len([e for e in events if 'stopped' in e]), 2)
        requests = [e['request'] for e in events if 'request' in e]
        self.assertEqual(len(requests), 8)
        for index, name in enumerate(('short', 'long')):
            payload = json.loads((ROOT / f'benchmarks/{name}.json').read_text())
            self.assertEqual(requests[index * 4:(index + 1) * 4], [payload] * 4)
            self.assertEqual(payload['max_tokens'], 512)
            self.assertFalse(payload['cache_prompt'])
            self.assertFalse(payload['chat_template_kwargs']['enable_thinking'])

    def test_failures_stop_owned_server_without_uploading_errors(self):
        for mode in ('http-error', 'cached', 'invalid'):
            with self.subTest(mode=mode):
                self.config.write_text(json.dumps({'events': str(self.events), 'mode': mode}))
                result = self.run_benchmark()
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.output.exists())
                self.assertNotIn('private-user', result.stderr)
                self.assertIn('stopped', self.read_events()[-1])

    def test_occupied_port_is_untouched(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', self.port))
            sock.listen()
            result = self.run_benchmark()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.events.exists())
        self.assertFalse(self.output.exists())

    def test_early_completion_keeps_actual_counts(self):
        self.config.write_text(json.dumps({'events': str(self.events), 'mode': 'early-stop'}))
        result = self.run_benchmark()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(self.output.read_text())
        for benchmark in report['benchmarks']:
            for run in [benchmark['warmup']] + benchmark['runs']:
                self.assertEqual(run['usage']['completion_tokens'], 64)
                self.assertEqual(run['timings']['predicted_n'], 64)
                self.assertEqual(run['finish_reason'], 'stop')

    def test_early_server_exit_fails_without_report(self):
        self.config.write_text(json.dumps({'events': str(self.events), 'mode': 'exit'}))
        result = self.run_benchmark()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('exited before becoming ready', result.stderr)
        self.assertFalse(self.output.exists())

    def test_interrupt_stops_server_without_report(self):
        self.config.write_text(json.dumps({'events': str(self.events), 'mode': 'slow'}))
        process = subprocess.Popen(self.command(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 10
            while not self.events.exists() or not any('request' in e for e in self.read_events()):
                self.assertLess(time.monotonic(), deadline, 'Runner did not send a request')
                time.sleep(0.05)
            process.send_signal(signal.SIGINT)
            _, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 130, stderr)
            self.assertIn('stopped', self.read_events()[-1])
            self.assertFalse(self.output.exists())
        finally:
            if process.poll() is None:
                process.terminate()
                process.communicate(timeout=15)

    def test_existing_output_is_not_overwritten(self):
        self.output.write_text('keep this')
        result = self.run_benchmark()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.output.read_text(), 'keep this')
        self.assertFalse(self.events.exists())


if __name__ == '__main__':
    if '--fake-server' in sys.argv:
        fake_server()
    else:
        unittest.main()
