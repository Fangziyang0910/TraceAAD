"""Launch twenty fresh runs with the previous batch's backend/seed mapping.

No retries, scheduler migration, old-run mutation, or automatic restart. The
manifest is written before any launch and updated after each successful spawn.
"""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path

from experiments.infra.base import REPO_ROOT, RESULTS_ROOT, TASK_SHORT
from experiments.infra.launcher import check_backends, is_session_alive, launch_command, write_json_atomic
from traceaad.v10_14_2 import Config


def build_plan(previous, batch):
    source = previous['plan']
    tasks = Counter(item['task'] for item in source)
    if len(source) != 20 or len(tasks) != 5 or set(tasks.values()) != {4}:
        raise ValueError('expected five tasks with four repeats in the previous batch')
    plan = []
    for item in source:
        task, repeat, backend, seed = (item[k] for k in ('task', 'repeat', 'backend', 'seed'))
        short = TASK_SHORT[task]
        name = f'{batch}_{short}_traceaad_v10_14_2_rep{repeat}'
        session = f'{batch}_{short}_r{repeat}'
        command = ['uv', 'run', 'python', '-m', 'experiments.traceaad_v10_14_2.run',
                   '--task', task, '--backend', backend, '--repeat', str(repeat),
                   '--seed', str(seed), '--run-name', name, '--budget=1000', '--eval-workers=4']
        plan.append({'task': task, 'repeat': repeat, 'seed': seed, 'backend': backend,
                     'session': session, 'run_name': name,
                     'run_dir': str(RESULTS_ROOT / 'traceaad_v10_14_2' / task / name),
                     'command': command, 'previous_run': item['run_dir'], 'status': 'planned'})
    if len({p['run_name'] for p in plan}) != 20:
        raise ValueError('duplicate planned run identity')
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-batch', type=Path, required=True)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not args.batch.replace('_', '').replace('-', '').isalnum():
        raise ValueError('batch must be an alphanumeric identity')
    previous = json.loads(args.from_batch.read_text())
    plan = build_plan(previous, args.batch)
    if args.dry_run:
        print(json.dumps({'batch': args.batch, 'plan': plan, 'policy': asdict(Config())}, indent=2))
        return
    root = RESULTS_ROOT / 'traceaad_v10_14_2'
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / f'batch_{args.batch}.json'
    if manifest_path.exists():
        raise ValueError('batch manifest already exists; inspect partial launches instead of retrying')
    for item in plan:
        if Path(item['run_dir']).exists() or is_session_alive(item['session']):
            raise ValueError(f'existing run/session: {item["run_name"]}')
    # Caller has stopped the previous batch. Do not silently share its slots.
    live_old = [p['session'] for p in previous['plan'] if is_session_alive(p['session'])]
    if live_old:
        raise RuntimeError(f'previous batch sessions still active: {live_old}')
    check_backends(p['backend'] for p in plan)
    files = sorted([*(REPO_ROOT / 'traceaad/v10_14_2').glob('*.py'),
                    *(REPO_ROOT / 'experiments/traceaad_v10_14_2').glob('*.py'),
                    REPO_ROOT / 'core/llm.py', REPO_ROOT / 'core/evaluate.py',
                    REPO_ROOT / 'traceaad/v10_13/storage.py', REPO_ROOT / 'traceaad/v10_13/parsing.py'])
    manifest = {'batch': args.batch, 'method': 'v1014_2', 'display_name': 'V10.14-2',
                'created_at': datetime.now().astimezone().isoformat(), 'status': 'launching',
                'previous_batch': str(args.from_batch), 'search_policy': asdict(Config()),
                'sampling': previous['sampling'], 'services': previous.get('services'),
                'target_distribution': dict(Counter(p['backend'] for p in plan)),
                'repeats': 4, 'eval_workers': 4, 'automatic_retries': False,
                'study': 'joint revision, not single-factor causal identification',
                'implementation_files': {str(p.relative_to(REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                'plan': plan}
    write_json_atomic(manifest_path, manifest)
    for item in plan:
        launch_command(item['session'], item['command'])
        item.update(status='started_unverified', started_at=datetime.now().astimezone().isoformat())
        write_json_atomic(manifest_path, manifest)
        print(item['session'], item['backend'], flush=True)
    manifest['status'] = 'started_pending_verification'
    write_json_atomic(manifest_path, manifest)
    print(manifest_path)


if __name__ == '__main__':
    main()
