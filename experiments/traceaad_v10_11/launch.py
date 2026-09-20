"""Queue five tasks with one to three seeds on free service slots."""

import argparse
from collections import Counter
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import re
import time

from experiments.infra.base import BACKEND_CAPACITY, BACKENDS, LaunchItem, TASKS, TASK_SHORT, free_slots, item_is_running, launch_items
from experiments.infra.launcher import check_backends, get_summary_status
from experiments.infra.equivalent_backends import prepare_resume
from llm4ad.method.traceaad_v10_11.storage import atomic_json

RESULTS_ROOT = Path(__file__).resolve().parent / 'results'
BACKEND_NAMES = tuple(BACKENDS)


def allocate(plan, available, backend_pool, *, defer_cvrp=False):
    remaining, assignments = dict(available), []
    used_by_task = {task: {row['backend'] for row in plan
                           if row['task'] == task and row['backend']}
                    for task in TASKS}
    for row in plan:
        if row['status'] != 'queued' or (defer_cvrp and row['task'] == 'cvrp_aco'):
            continue
        candidates = [backend for backend in backend_pool if remaining.get(backend, 0) > 0]
        if not candidates:
            continue
        used = used_by_task[row['task']]
        candidates.sort(key=lambda backend: (
            backend in used,
            (BACKEND_CAPACITY[backend] - remaining[backend] + 1) / BACKEND_CAPACITY[backend],
            BACKEND_NAMES.index(backend)))
        backend = candidates[0]
        remaining[backend] -= 1
        used.add(backend)
        assignments.append((row, backend))
    return assignments


def healthy_slots(available, backend_pool):
    available = dict(available)
    for backend in backend_pool:
        if available.get(backend, 0) > 0:
            try:
                check_backends([backend])
            except Exception:
                available[backend] = 0
    return available


def build_plan(batch, prefix, thinking=False, history_code=False, repeats=3, traj_gens=8,
               cvrp_last=False, rand_context=False, n_references=8):
    order = [(repeat, task) for repeat in range(1, repeats + 1) for task in TASKS]
    if cvrp_last:
        order.sort(key=lambda item: item[1] == 'cvrp_aco')
    return [dict(task=task, repeat=repeat, seed=repeat-1, backend=None,
                 run_name=f'{batch}_{TASK_SHORT[task]}_v1011_rep{repeat}',
                 session=f'{prefix}_{TASK_SHORT[task]}_r{repeat}', attempts=0, status='queued',
                 traj_gens=0 if rand_context else traj_gens,
                 **({'thinking': True} if thinking else {}),
                 **({'history_code': True} if history_code else {}),
                 **({'rand_context': True, 'n_references': n_references} if rand_context else {}))
            for repeat, task in order]


def cvrp_group_finished(batch):
    manifest = RESULTS_ROOT / f'batch_{batch}.json'
    if not manifest.exists():
        return False
    rows = [row for row in json.loads(manifest.read_text())['plan']
            if row['task'] == 'cvrp_aco']
    return bool(rows) and all(get_summary_status(launch_item(row).run_dir) == 'finished'
                              for row in rows)


def launch_item(row):
    flags = [flag for flag, enabled in (
        ('--thinking', row.get('thinking')),
        ('--history-code', row.get('history_code')),
    ) if enabled]
    if row.get('rand_context'):
        flags += ['--rand-context', '--n-references', str(row.get('n_references', 8))]
    else:
        flags += ['--traj-gens', str(row.get('traj_gens', 8))]
    return LaunchItem(task=row['task'], repeat=row['repeat'], seed=row['seed'],
                      backend=row['backend'], session=row['session'], run_name=row['run_name'],
                      run_dir=RESULTS_ROOT / row['task'] / row['run_name'],
                      module='experiments.traceaad_v10_11.run',
                      extra_args=tuple(flags))


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


def verify_runtime():
    root = Path(__file__).resolve().parents[2]
    manifest = root / 'runtime_manifest.json'
    if not manifest.exists():
        raise ValueError('freeze the reviewed source first using experiments.traceaad_v10_11.freeze')
    payload = json.loads(manifest.read_text())
    for relative, expected in payload['files'].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'frozen source changed: {relative}')
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--session-prefix', default='v1011')
    parser.add_argument('--repeats', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--traj-gens', type=int, default=8,
                        help='number of formation-history steps in context; 0 omits the history block')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--backends', default=','.join(BACKEND_NAMES),
                        help='comma-separated backend pool for this batch')
    parser.add_argument('--direct', action='store_true',
                        help='start all currently available assignments in each poll')
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--max-attempts', type=int, default=3)
    parser.add_argument('--thinking', action='store_true',
                        help='stamp every run of this batch with model thinking mode')
    parser.add_argument('--history-code', action='store_true',
                        help='include historical programs in each formation path')
    parser.add_argument('--rand-context', action='store_true',
                        help='replace formation history with rank-sampled archive references')
    parser.add_argument('--n-references', type=int, default=8,
                        help='number of archive reference cards in random-context mode')
    parser.add_argument('--cvrp-last', action='store_true',
                        help='queue all CVRP repeats after the other twelve runs')
    parser.add_argument('--cvrp-barrier-batches', default='',
                        help='defer CVRP until at least one named batch has finished all CVRP repeats')
    args = parser.parse_args(argv)
    if args.interval < 1 or args.max_attempts < 1:
        parser.error('interval and max-attempts must be positive')
    if args.traj_gens < 0:
        parser.error('traj-gens must be nonnegative')
    if args.n_references < 1:
        parser.error('n-references must be positive')
    if args.rand_context and (args.history_code or args.traj_gens != 8):
        parser.error('--rand-context excludes --history-code and non-default --traj-gens')
    if not args.rand_context and args.n_references != 8:
        parser.error('--n-references requires --rand-context')
    if not all(re.fullmatch(r'[A-Za-z0-9_-]+', s) for s in (args.batch, args.session_prefix)):
        parser.error('batch and prefix must contain only letters, numbers, underscore or hyphen')
    cvrp_barriers = tuple(filter(None, args.cvrp_barrier_batches.split(',')))
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', batch) for batch in cvrp_barriers):
        parser.error('CVRP barrier batch names must contain only letters, numbers, underscore or hyphen')
    backend_pool = tuple(dict.fromkeys(args.backends.split(',')))
    if not backend_pool or any(backend not in BACKENDS for backend in backend_pool):
        parser.error(f'backends must be drawn from {", ".join(BACKEND_NAMES)}')
    identity = None if args.dry_run else verify_runtime()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = RESULTS_ROOT / f'batch_{args.batch}.json'
    with manifest.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if manifest.exists():
            payload = json.loads(manifest.read_text())
            if (payload['method'], payload['session_prefix']) != ('v1011', args.session_prefix):
                raise ValueError('batch identity mismatch')
            if tuple(payload.get('backends', ())) != backend_pool:
                raise ValueError('batch backend pool mismatch')
            if bool(payload.get('history_code')) != args.history_code:
                raise ValueError('batch history-code mismatch')
            if (payload.get('repeats', 3), payload.get('traj_gens', 8)) != (args.repeats, args.traj_gens):
                raise ValueError('batch repeats or history-length mismatch')
            if (bool(payload.get('rand_context')), payload.get('n_references', 8)) != (
                    args.rand_context, args.n_references):
                raise ValueError('batch random-context mismatch')
            if (bool(payload.get('cvrp_last')), tuple(payload.get('cvrp_barriers', ()))) != (args.cvrp_last, cvrp_barriers):
                raise ValueError('batch CVRP scheduling mismatch')
            if not args.dry_run and payload['source_identity'] != identity:
                raise ValueError('batch frozen source mismatch')
        else:
            payload = dict(method='v1011', batch=args.batch, session_prefix=args.session_prefix,
                           source_identity=identity, created_at=datetime.now().astimezone().isoformat(),
                           thinking=args.thinking, history_code=args.history_code,
                           repeats=args.repeats, traj_gens=args.traj_gens,
                           rand_context=args.rand_context, n_references=args.n_references,
                           cvrp_last=args.cvrp_last, cvrp_barriers=cvrp_barriers,
                           backends=backend_pool, direct=args.direct,
                           plan=build_plan(args.batch, args.session_prefix, args.thinking,
                                           args.history_code, args.repeats, args.traj_gens,
                                           args.cvrp_last, args.rand_context, args.n_references))
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
                available = healthy_slots(available, backend_pool)
            defer_cvrp = bool(cvrp_barriers) and not any(
                cvrp_group_finished(batch) for batch in cvrp_barriers)
            assignments = allocate(plan, available, backend_pool,
                                   defer_cvrp=defer_cvrp)
            if not args.direct:
                assignments = assignments[:1]
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
