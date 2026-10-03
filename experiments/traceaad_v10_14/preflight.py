"""Paid, isolated three-candidate smoke checks on five small real tasks.

The last candidate is deliberately an independent Pivot to verify production
requests and bookkeeping before spending the formal twenty-run budget.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import time

from experiments.infra.base import BACKENDS, build_llm_client
from benchmarks.cvrp_aco import CVRPACOEvaluation
from benchmarks.online_bin_packing import OBPEvaluation
from benchmarks.op_aco import OPACOEvaluation
from benchmarks.tsp_construct import TSPEvaluation
from benchmarks.vrptw_construct import VRPTWEvaluation
from traceaad.v10_14 import Config, TraceAADV1014


ROUTES = [('tsp_construct', 'local'), ('online_bin_packing', 'server3b'),
          ('vrptw_construct', 'server3'), ('cvrp_aco', 'server3'), ('op_aco', 'server3b')]


def small_task(task, seed=10):
    if task == "tsp_construct":
        return TSPEvaluation(n_instance=2, problem_size=10, seed=seed)
    if task == "vrptw_construct":
        return VRPTWEvaluation(n_instance=2, problem_size=10, seed=seed)
    if task == "online_bin_packing":
        return OBPEvaluation(dataset_specs=[{"n_instances": 1, "n_items": 64, "capacities": [100, 500]}], seed=seed)
    cls = CVRPACOEvaluation if task == "cvrp_aco" else OPACOEvaluation
    evaluation = cls(split="train" if seed == 10 else "val_50", n_ants=3, n_iterations=2, n_workers=1)
    evaluation._datasets = evaluation._datasets[:2]
    evaluation.n_instance = 2
    return evaluation


def run_one(output, task, backend):
    profile = BACKENDS[backend]
    llm = build_llm_client(base_url=profile.base_url, model=profile.model,
                          no_proxy='127.0.0.1,localhost,222.201.145.6', max_tokens=8192)
    llm._client = llm._client.with_options(max_retries=0)
    config = Config(budget=3, max_evaluations=3, init_proposals=1)
    directory = output / task
    if directory.exists():
        raise ValueError('preflight run already exists; do not hide retries')
    directory.mkdir(parents=True)
    (directory / 'preflight_config.json').write_text(json.dumps({
        'backend': backend, 'task': task, 'config': asdict(config),
        'scope': 'small real evaluator smoke; not formal performance',
        'forced_main_action': 'Pivot'}, indent=2))
    started = time.monotonic()
    try:
        m = TraceAADV1014(evaluation=small_task(task), selection_evaluation=small_task(task, seed=11),
                           llm=llm, task=task, config=config, run_dir=directory)
        m._contract = lambda anchor: ('Pivot', 'None', None)
        result = m.run()
        attempts = list(m.facts.tables['attempt'].values())
        requests = list(m.facts.tables['request'].values())
        checks = {
            'three_paid_attempts': m.ledger.candidates == 3,
            'ordinary_text_requests': all(r['output_mode'] == 'full' and not r.get('response_format') and not r.get('structured_outputs') for r in requests),
            'idea_accounted': all(a.get('delivery', {}).get('idea_status') in ('present', 'missing_in_response') for a in attempts if a['status'] in ('ok', 'duplicate', 'evaluation_failed')),
            'all_deliveries_parsed': len(attempts) == 3 and all(a['status'] in ('ok', 'duplicate', 'evaluation_failed') for a in attempts),
            'independent_pivot_executed': any(a.get('context_mode') == 'independent' and a['parent_id'] is None for a in attempts),
            'valid_candidate_and_selection': bool(m.anchors) and result['status'] == 'finished',
        }
        report = {'task': task, 'backend': backend, 'checks': checks, 'passed': all(checks.values()),
                  'seconds': time.monotonic()-started, 'tokens': m.ledger.tokens,
                  'idea_status': [a.get('delivery', {}).get('idea_status') for a in attempts],
                  'attempts': [{k: a.get(k) for k in ('id', 'status', 'error', 'context_mode', 'parent_id')} for a in attempts]}
    except Exception as exc:
        report = {'task': task, 'backend': backend, 'passed': False, 'error': str(exc)}
    finally:
        llm.close()
    (directory / 'verification.json').write_text(json.dumps(report, indent=2))
    print(task, backend, report, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', choices=[task for task, _ in ROUTES])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        routes = [route for route in ROUTES if not args.tasks or route[0] in args.tasks]
        results = list(pool.map(lambda route: run_one(args.output, *route), routes))
    (args.output / 'verification.json').write_text(json.dumps(results, indent=2))
    if not all(r['passed'] for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
