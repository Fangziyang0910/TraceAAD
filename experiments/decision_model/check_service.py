"""Check the real model's authenticated HTTP service without generating a candidate."""

import argparse
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from data import ACTIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter")
    parser.add_argument("metadata", type=Path)
    parser.add_argument("paired_data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output file")
    metadata = json.loads(args.metadata.read_text())
    assert metadata["decision_horizon"] == 2
    rows = list(map(json.loads, (args.paired_data / 'calibration.jsonl').read_text().splitlines()))
    first = rows[0]
    states = [r['state'] for r in rows if r['run_id'] == first['run_id'] and
              r['state_id'].split('/')[0] == first['state_id'].split('/')[0]]
    assert len(states) == 3 and {s['proposed_action'] for s in states} == set(ACTIONS)
    with socket.socket() as available:
        available.bind(('127.0.0.1', 8813))
    token_file = args.output.with_suffix('.token')
    opener = build_opener(ProxyHandler({}))
    log_path = args.output.with_suffix('.log')
    with log_path.open('w') as log:
        server = subprocess.Popen([sys.executable, '-u', str(Path(__file__).with_name('serve.py')),
            args.adapter, str(args.metadata), '--token-file', str(token_file), '--load-in-4bit'],
            stdout=log, stderr=subprocess.STDOUT)
        try:
            started = time.monotonic()
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'HTTP service exited; see {log_path}')
                try:
                    headers = {'Authorization': 'Bearer ' + token_file.read_text().strip(), 'Content-Type': 'application/json'}
                    with opener.open(Request('http://127.0.0.1:8813/metadata', headers=headers), timeout=2) as response:
                        actual = json.load(response)
                    break
                except (OSError, URLError):
                    if time.monotonic() - started > 180:
                        raise RuntimeError(f'HTTP service did not become ready; see {log_path}')
                    time.sleep(2)
            for key in ('revision', 'prompt_policy', 'prompt_sources_sha256', 'score_scales', 'decision_horizon'):
                assert actual[key] == metadata[key], key
            try:
                opener.open('http://127.0.0.1:8813/metadata', timeout=2)
                raise AssertionError('unauthenticated request was accepted')
            except HTTPError as error:
                assert error.code == 401
            body = json.dumps({'records': states}).encode()
            with opener.open(Request('http://127.0.0.1:8813/score', data=body, headers=headers), timeout=120) as response:
                choices = json.load(response)['choices']
            assert len(choices) == 3 and {c['action'] for c in choices} == set(ACTIONS)
            for choice in choices:
                assert math.isfinite(choice['expected_gain']) and choice['expected_gain'] >= 0
                assert 1 <= choice['expected_candidate_cost'] <= 2
                assert math.isclose(choice['gain_per_candidate'], choice['expected_gain'] / choice['expected_candidate_cost'], rel_tol=1e-9, abs_tol=1e-12)
                assert set(choice['answers']) == {'frontier_gain', 'repair_used'}
            report = {'native_http_validated': True, 'unauthenticated_status': 401,
                      'scored_actions': [c['action'] for c in choices], 'generation_candidates_used': 0,
                      'test_metrics_used': False}
        finally:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=20)
    assert server.poll() is not None
    report['service_stopped_after_check'] = True
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
