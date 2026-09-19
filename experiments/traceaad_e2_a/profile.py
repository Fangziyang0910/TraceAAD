"""Profile the complete historical archive needed by E2-A trajectory sensors."""
import argparse
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from experiments.traceaad_refine_e1.profile import candidate, worker
from experiments.traceaad_refine_e1 import profile_core
from experiments.traceaad_refine_e1.prepare import dump
from .prepare import DEFAULT, ROOT

PRIMARY_TASKS = {'tsp_construct', 'online_bin_packing', 'vrptw_construct'}
E1_PROFILES = ROOT / 'experiments/traceaad_refine_e1/raw/refine_e1_20260907/profiles'


def inputs(out=DEFAULT):
    manifest = json.loads((out / 'snapshot.json').read_text())
    result = []
    for run in manifest['runs']:
        if run['task'] not in PRIMARY_TASKS:
            continue
        nodes = json.loads((out / 'snapshot' / run['run_name'] / 'tree_state.json').read_text())['nodes']
        result.append((run, nodes))
    return result


def profile(out=DEFAULT, workers=12):
    jobs = {}
    reused = 0
    for run, nodes in inputs(out):
        for node in nodes:
            key = candidate(node)['key']
            target = out / 'profiles' / run['task'] / f'{key}.json'
            if target.exists():
                continue
            old = E1_PROFILES / run['task'] / f'{key}.json'
            if old.exists() and all(v['ok'] for v in json.loads(old.read_text())['panels'].values()):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(old, target)
                reused += 1
            else:
                jobs[(run['task'], key)] = (run['task'], node)
    print('reused', reused, 'new jobs', len(jobs), 'workers', workers, flush=True)
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, job): key for key, job in jobs.items()}
        for i, future in enumerate(as_completed(futures), 1):
            task, key = futures[future]
            result = future.result()
            dump(out / 'profiles' / task / f'{key}.json', result)
            if i == 1 or i % 25 == 0:
                print('profiles', i, '/', len(jobs), 'seconds', round(time.time() - started), flush=True)
    matrices(out, reused, len(jobs), time.time() - started)


def matrices(out=DEFAULT, reused=0, new_jobs=0, elapsed=0.0):
    coverage = []
    for run, nodes in inputs(out):
        valid = []
        panels = {'A': [], 'B': []}
        for node in nodes:
            path = out / 'profiles' / run['task'] / f"{candidate(node)['key']}.json"
            record = json.loads(path.read_text())
            if all(record['panels'][panel]['ok'] for panel in panels):
                valid.append(node['id'])
                for panel in panels:
                    panels[panel].append(record['panels'][panel])
        prefix = run['task'] in profile_core.PREFIX_TASKS
        values = {panel: profile_core.compute_distance_matrix(records, prefix_mode=prefix)
                  for panel, records in panels.items()}
        folder = out / 'distances' / run['run_name']
        folder.mkdir(parents=True, exist_ok=True)
        dump(folder / 'ids.json', valid)
        np.save(folder / 'behavior.npy', (values['A'] + values['B']) / 2)
        coverage.append({'run': run['run_name'], 'task': run['task'], 'nodes': len(nodes), 'valid_profiles': len(valid)})
        print('matrix', run['run_name'], len(valid), '/', len(nodes), flush=True)
    dump(out / 'profile_coverage.json', {
        'runs': coverage,
        'reused_success_profiles': reused,
        'new_profile_jobs': new_jobs,
        'new_profile_wall_seconds': elapsed,
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--out', type=Path, default=DEFAULT)
    args = parser.parse_args()
    profile(args.out, args.workers)
