"""Queue five tasks x three seeds; start at most one run per poll on a free service slot."""

import argparse
from collections import Counter
from datetime import datetime
import fcntl
import json
from pathlib import Path
import re
import time

from experiments.infra.equivalent_backends import prepare_resume
from experiments.infra.base import BACKENDS, LaunchItem, TASKS, TASK_SHORT, free_slots, item_is_running, launch_items
from experiments.infra.launcher import check_backends, get_summary_status


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2) + '\n')
    tmp.replace(path)


def healthy_slots(available: dict[str, int], backend_pool: tuple[str, ...]) -> dict[str, int]:
    healthy = dict(available)
    for backend in backend_pool:
        if healthy.get(backend, 0) > 0:
            try:
                check_backends([backend])
            except Exception:
                healthy[backend] = 0
    return healthy


def allocate(plan: list[dict], available: dict[str, int], backend_pool: tuple[str, ...] = BACKENDS) -> list[tuple[dict, str]]:
    remaining = dict(available)
    assignments = []
    used_by_task = {task: {r['backend'] for r in plan if r['task'] == task and r['backend']}
                    for task in TASKS}
    for row in plan:
        if row['status'] != 'queued':
            continue
        preferred = [b for b in backend_pool if remaining.get(b, 0) > 0 and b not in used_by_task[row['task']]]
        if not preferred:
            preferred = [b for b in backend_pool if remaining.get(b, 0) > 0]
        if preferred:
            backend = preferred[0]
            remaining[backend] -= 1
            used_by_task[row['task']].add(backend)
            assignments.append((row, backend))
    return assignments


RESULTS_ROOT = Path(__file__).resolve().parent / 'results'


def build_plan(batch, prefix, thinking=False):
    return [dict(task=task, repeat=repeat, seed=repeat-1, backend=None,
                 run_name=f'{batch}_{TASK_SHORT[task]}_v1010_rep{repeat}',
                 session=f'{prefix}_{TASK_SHORT[task]}_r{repeat}', attempts=0, status='queued',
                 **({'thinking': True} if thinking else {}))
            for repeat in range(1, 4) for task in TASKS]


def launch_item(row):
    return LaunchItem(task=row['task'], repeat=row['repeat'], seed=row['seed'],
                      backend=row['backend'], session=row['session'], run_name=row['run_name'],
                      run_dir=RESULTS_ROOT / row['task'] / row['run_name'],
                      module='experiments.traceaad_v10_10.run',
                      extra_args=('--thinking',) if row.get('thinking') else ())


def refresh(plan, max_attempts):
    for row in plan:
        item = launch_item(row)
        status = get_summary_status(item.run_dir)
        if item_is_running(item):
            row['status'] = 'running'
        elif status in ('finished', 'blocked'):
            row['status'] = status
        elif row['attempts'] >= max_attempts:
            row['status'] = 'stopped'
        else:
            row['status'] = 'queued'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--session-prefix', default='v1010')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--max-attempts', type=int, default=3)
    parser.add_argument('--thinking', action='store_true',
                        help='stamp every run of this batch with model thinking mode')
    args = parser.parse_args()
    if args.interval < 1 or args.max_attempts < 1:
        parser.error('interval and max-attempts must be positive')
    if not all(re.fullmatch(r'[A-Za-z0-9_-]+', s) for s in (args.batch, args.session_prefix)):
        parser.error('batch and prefix must contain only letters, numbers, underscore or hyphen')
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = RESULTS_ROOT / f'batch_{args.batch}.json'
    with manifest.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if manifest.exists():
            payload = json.loads(manifest.read_text())
            if (payload['method'], payload['session_prefix']) != ('v1010', args.session_prefix):
                raise ValueError('batch identity mismatch')
        else:
            payload = dict(method='v1010', batch=args.batch, session_prefix=args.session_prefix,
                           created_at=datetime.now().astimezone().isoformat(),
                           thinking=args.thinking,
                           plan=build_plan(args.batch, args.session_prefix, args.thinking))
            if any(item_is_running(launch_item(r)) or launch_item(r).run_dir.exists()
                   for r in payload['plan']):
                raise ValueError('existing session or run directory without matching batch manifest')
        last = None
        while True:
            plan = payload['plan']
            refresh(plan, args.max_attempts)
            available = free_slots()
            if not args.dry_run:
                # Recheck service health on every actual start; no stale health cache.
                available = healthy_slots(available, set())
            assignments = allocate(plan, available)[:1]
            if args.dry_run:
                print(json.dumps(dict(plan=plan, free=available, next=[b for _, b in assignments]), indent=2))
                return
            for row, backend in assignments:
                # A competing legacy watcher may have occupied the slot during health checks.
                if free_slots().get(backend, 0) <= 0:
                    continue
                row.update(backend=backend, attempts=row['attempts']+1, status='launching')
                atomic_json(manifest, payload)
                try:
                    prepare_resume(launch_item(row), BACKENDS)
                    launch_items([launch_item(row)], dry_run=False)
                except Exception as exc:
                    row.update(status='queued', last_launch_error=str(exc))
                else:
                    row['status'] = 'running'
            payload['updated_at'] = datetime.now().astimezone().isoformat()
            atomic_json(manifest, payload)
            counts = dict(Counter(r['status'] for r in plan))
            if counts != last:
                print(f'{payload["updated_at"]} {counts} manifest={manifest}', flush=True)
                last = counts
            if not args.watch or all(r['status'] in ('finished', 'blocked', 'stopped') for r in plan):
                return
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
