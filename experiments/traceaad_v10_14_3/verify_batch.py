"""Read actual requests, completed candidates, process identities and source hashes."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from experiments.infra.base import REPO_ROOT
from experiments.infra.launcher import is_session_alive, write_json_atomic


def verify(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    hashes_ok = all(hashlib.sha256((REPO_ROOT / p).read_bytes()).hexdigest() == value
                    for p, value in manifest['implementation_files'].items())
    rows = []
    for item in manifest['plan']:
        directory = Path(item['run_dir'])
        row = {k: item[k] for k in ('task', 'repeat', 'backend', 'run_name')}
        row['session_alive'] = is_session_alive(item['session'])
        if not (directory / 'search.jsonl').exists():
            rows.append({**row, 'ready': False, 'reason': 'journal not yet created'})
            continue
        # Read only complete lines from this instantaneous prefix.
        with (directory / 'search.jsonl').open('rb') as f:
            raw = f.read()
        records = [json.loads(line) for line in raw.splitlines(keepends=True) if line.endswith(b'\n')]
        attempts = [r['data'] for r in records if r['kind'] == 'attempt']
        requests = [r['data'] for r in records if r['kind'] == 'request']
        states = [r['state'] for r in records if 'state' in r]
        state = states[-1] if states else {}
        valid = [a for a in attempts if a['status'] in ('ok', 'duplicate')]
        cfg = json.loads((directory / 'run_config.json').read_text())
        previous = json.loads((Path(item['previous_run']) / 'run_config.json').read_text())
        ordinary_requests = all(r.get('output_mode') == 'full' and not r.get('response_format')
                                and not r.get('structured_outputs')
                                and not r.get('model_config', {}).get('extra_body', {}).get('structured_outputs')
                                for r in requests)
        idea_accounted = all(a.get('delivery', {}).get('idea_status') ==
                             ('present' if a.get('idea', '').strip() else 'missing_in_response') for a in valid)
        settings = state.get('identity', {}).get('config', {})
        same_evaluation = cfg['task_eval'] == previous['task_eval']
        same_sampling = all(cfg['llm'].get(k) == previous['llm'].get(k) for k in
                            ('base_url', 'model', 'temperature', 'top_p', 'top_k', 'max_tokens', 'enable_thinking', 'chars_per_token'))
        row.update(completed=len(attempts), status=dict(Counter(a['status'] for a in attempts)),
                   valid_candidates=len(valid), valid_empty_ideas=sum(not a.get('idea', '').strip() for a in valid),
                   phase=state.get('phase'), ledger=state.get('ledger'), pending=state.get('pending'),
                   request_count=len(requests), ordinary_text_requests=ordinary_requests, idea_accounted=idea_accounted,
                   same_evaluation=same_evaluation, same_sampling=same_sampling,
                   independent_pivots=sum(a.get('context_mode') == 'independent' for a in attempts),
                   formal_policy=(settings.get('budget') == 1000 and settings.get('output_mode') == 'full'
                                  and settings.get('parent_policy') == 'rank_count' and settings.get('pivot_context') == 'independent'))
        row['ready'] = (row['session_alive'] and bool(valid) and bool(requests) and idea_accounted
                        and ordinary_requests and same_evaluation and same_sampling and row['formal_policy'])
        rows.append(row)
    report = {'checked_at': datetime.now(timezone.utc).isoformat(), 'implementation_hashes_match': hashes_ok,
              'all_ready': hashes_ok and len(rows) == 20 and all(r['ready'] for r in rows), 'runs': rows}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mark-running', action='store_true')
    args = parser.parse_args()
    report = verify(args.manifest)
    write_json_atomic(args.output, report)
    print(json.dumps({
        'checked_at': report['checked_at'], 'hashes_match': report['implementation_hashes_match'],
        'all_ready': report['all_ready'], 'ready_runs': sum(r['ready'] for r in report['runs']),
        'completed': sum(r.get('completed', 0) for r in report['runs']),
        'status': dict(sum((Counter(r.get('status', {})) for r in report['runs']), Counter())),
        'not_ready': [r for r in report['runs'] if not r['ready']],
    }, ensure_ascii=False))
    if args.mark_running:
        if not report['all_ready']:
            raise SystemExit('startup checks are not complete')
        manifest = json.loads(args.manifest.read_text())
        manifest.update(status='running', verified_at=report['checked_at'], startup_report=str(args.output))
        for item in manifest['plan']:
            item['status'] = 'running_verified'
        write_json_atomic(args.manifest, manifest)


if __name__ == '__main__':
    main()
