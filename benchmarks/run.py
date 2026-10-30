#!/usr/bin/env python3
"""Run fixed local inference requests; collect benchmark metrics only."""

import argparse
from contextlib import contextmanager
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import time

PAYLOADS = Path(__file__).resolve().parent
TIMINGS = ('cache_n', 'prompt_n', 'prompt_ms', 'prompt_per_second',
           'predicted_n', 'predicted_ms', 'predicted_per_second')


def request(port, path, payload=None, timeout=600):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=timeout)
    try:
        connection.request('GET' if payload is None else 'POST', path, body=payload,
                           headers={'Content-Type': 'application/json'})
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError(f'Server returned HTTP {response.status}; see the local server output.')
        return json.loads(response.read())
    finally:
        connection.close()


@contextmanager
def server(args):
    # Refuse an existing listener before starting or sending any HTTP request.
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', args.port))
    command = [args.binary]
    if args.config:
        command += ['--config', args.config]
    command += ['--port', str(args.port)]
    process = subprocess.Popen(command, start_new_session=True)
    try:
        deadline = time.monotonic() + 60
        while True:
            if process.poll() is not None:
                raise ValueError('MoEShift exited before becoming ready; see the local server output.')
            try:
                if request(args.port, '/health', timeout=1).get('status') == 'ok':
                    break
            except (OSError, ValueError, http.client.HTTPException):
                pass
            if time.monotonic() >= deadline:
                raise ValueError('Timed out waiting for MoEShift to become ready.')
            time.sleep(0.1)
        yield
    finally:
        # The new session contains only our router and its worker.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def measure(port, payload):
    start = time.perf_counter()
    response = request(port, '/v1/chat/completions', payload)
    wall_ms = (time.perf_counter() - start) * 1000
    try:
        timings = {key: response['timings'][key] for key in TIMINGS}
        usage = {key: response['usage'][key] for key in ('prompt_tokens', 'completion_tokens')}
        finish = response['choices'][0]['finish_reason']
        for key, value in {**timings, **usage}.items():
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('Non-numeric benchmark metric.')
            if (key == 'cache_n' and value != 0) or (key != 'cache_n' and value <= 0):
                raise ValueError('Cached prompt or nonpositive benchmark metric.')
        if usage['completion_tokens'] > 512 or timings['predicted_n'] > 512:
            raise ValueError('Output exceeded the benchmark limit.')
        if finish not in ('stop', 'length'):
            raise ValueError('Unexpected completion finish reason.')
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError('Missing or invalid benchmark metrics in server response.') from error
    return {'wall_ms': wall_ms, 'inference_ms': timings['prompt_ms'] + timings['predicted_ms'],
            'timings': timings, 'usage': usage, 'finish_reason': finish}


def run(args):
    if args.output.exists():
        raise ValueError('Output already exists; choose a new --output filename.')
    report = {'schema_version': 1, 'protocol': 'moeshift-community-v1', 'benchmarks': []}
    for name in ('short', 'long'):
        payload = (PAYLOADS / f'{name}.json').read_bytes()
        with server(args):
            print(f'{name}: warmup (includes model loading)', flush=True)
            warmup = measure(args.port, payload)
            runs = []
            for index in range(3):
                print(f'{name}: measured run {index + 1}/3', flush=True)
                runs.append(measure(args.port, payload))
        median = {key: statistics.median(run[key] for run in runs)
                  for key in ('wall_ms', 'inference_ms')}
        median.update({key: statistics.median(run['timings'][key] for run in runs)
                       for key in ('prompt_ms', 'predicted_ms', 'prompt_per_second', 'predicted_per_second')})
        report['benchmarks'].append({'name': name, 'payload_sha256': hashlib.sha256(payload).hexdigest(),
                                     'warmup': warmup, 'runs': runs, 'median': median})
    # Only allowlisted benchmark fields are saved, never raw responses or config.
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write('\n')
    print('Benchmark complete. Review the JSON before attaching it to an issue.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', default='moeshift', help='MoEShift executable (default: moeshift on PATH)')
    parser.add_argument('--config', help='Optional MoEShift JSON configuration; otherwise normal discovery applies')
    parser.add_argument('--port', type=int, default=8080, help='Local server port (default: 8080)')
    parser.add_argument('--output', type=Path, default=PAYLOADS / 'benchmark-results.json')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')

    def interrupt(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        run(args)
    except KeyboardInterrupt:
        print('Benchmark interrupted; no report was written.', file=sys.stderr)
        return 130
    except (OSError, ValueError, http.client.HTTPException) as error:
        # HTTP error bodies may contain private paths: never print or save them.
        print(f'Benchmark failed: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
