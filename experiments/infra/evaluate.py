"""Evaluate committed programs with the same task and execution path as search.

    uv run python -m experiments.infra.evaluate RUN_DIR [...] --units 50,100,200

All methods use the same per-instance execution for the six-task suite.
Existing scores are returned unchanged when the frozen program matches.
"""

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.tasks import ALL_TASKS, PRIMARY_SPLITS, SPLITS, heldout_task, scale_of_split, split_of_scale
from traceaad.common.config import REVISION
from core.evaluate import EVALUATION_SEED
from .evaluation_execution import heldout_evaluator
from traceaad.common.storage import heldout_identity, read_json, save_heldout, stored_heldout, write_json
from .artifacts import load_run_summary, pick_best_sample


def parse_units(task, text=None):
    if text is None:
        return list(SPLITS[task])
    units = [unit.strip() for unit in text.split(',') if unit.strip()]
    splits = [unit if unit in SPLITS[task] or unit == 'eval' else split_of_scale(task, unit)
              for unit in units]
    if not splits or any(split not in SPLITS[task] and split != 'eval' for split in splits):
        raise ValueError(f'unknown {task} units: {text}')
    return splits


def evaluate_run(run_dir, splits, *, condition='shared', workers=None, timeout_seconds=None,
                 sample_order=None, max_sample_order=None, allow_incomplete=False,
                 variant=None, scheduler_socket=None):
    run_dir = Path(run_dir)
    config = read_json(run_dir / 'run_config.json')
    task = config['task']
    chosen, _ = pick_best_sample(run_dir, sample_order=sample_order,
        max_sample_order=max_sample_order, allow_incomplete=allow_incomplete)
    summary = load_run_summary(run_dir, require_finished=not allow_incomplete)
    code = chosen['program']
    key = hashlib.sha256(code.encode()).hexdigest()
    seeds = tuple(config.get('method_params', {}).get('evaluation_seeds', [EVALUATION_SEED]))
    results = []
    for split in splits:
        result = {'task': task, 'split': split, 'scale': str(scale_of_split(task, split)),
                  'key': key, 'node_id': chosen['node_id'], 'verification': 'verified'}
        result['verification'] = heldout_identity(run_dir, result)
        previous = stored_heldout(run_dir, result['scale'], variant)
        if previous:
            if (previous['key'], previous.get('node_id')) != (key, chosen['node_id']):
                raise ValueError('existing result belongs to another program')
            results.append({**previous, "verification": result["verification"]})
            continue
        evaluator, execution = heldout_evaluator(config, split, seeds, workers=workers,
            timeout_seconds=timeout_seconds, measure_calls=True,
            task_factory=heldout_task, scheduler_socket=scheduler_socket)
        outcome = evaluator.evaluate(code, key)
        records = outcome['evaluations']
        score = outcome['fitness']
        result.update(condition=condition, revision=REVISION, protocol=evaluator.protocol,
                      evaluation_seeds=list(seeds), **execution, fitness=score, evaluations=records,
                      objective=score,
                      seconds=sum(row['seconds'] for row in records), scores=[row['score'] for row in records if row['valid']],
                      failures=[{'kind': row['failure_kind'], 'error': row['error']} for row in records if not row['valid']],
                      search_status=summary['status'], sample_order=chosen['sample_order'],
                      variant="" if variant is None else variant)
        save_heldout(run_dir, result, result['variant'])
        results.append(result)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dirs', nargs='+', type=Path)
    parser.add_argument('--task', choices=ALL_TASKS)
    parser.add_argument('--units')
    parser.add_argument('--primary', action='store_true', help='only the six-task suite primary same-scale test')
    parser.add_argument('--timeout', type=float)
    parser.add_argument('--workers', type=int, help='instance workers; defaults to the saved execution settings')
    parser.add_argument('--scheduler-socket', help='override saved CPU scheduler; empty string disables it')
    parser.add_argument('--condition', choices=('shared', 'traceaad'), default='shared')
    parser.add_argument('--variant', help='explicit name for an additional evaluation')
    parser.add_argument('--sample-order', type=int)
    parser.add_argument('--max-sample-order', type=int)
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if args.primary and args.units:
        raise ValueError('--primary and --units are alternatives')
    if args.sample_order is not None and len(args.run_dirs) != 1:
        raise ValueError('--sample-order requires one run directory')
    payload = []
    for run in args.run_dirs:
        task = read_json(run / 'run_config.json')['task']
        if args.task and task != args.task:
            raise ValueError(f'{run}: task is {task}, not {args.task}')
        if args.primary and task not in PRIMARY_SPLITS:
            raise ValueError(f'{task} is not in the six-task suite')
        results = evaluate_run(run, list(PRIMARY_SPLITS[task]) if args.primary else parse_units(task, args.units), condition=args.condition,
            workers=args.workers, timeout_seconds=args.timeout, sample_order=args.sample_order,
            max_sample_order=args.max_sample_order, allow_incomplete=args.allow_incomplete,
            variant=args.variant, scheduler_socket=args.scheduler_socket)
        payload.extend({'run_dir': str(run), **result} for result in results)
    if args.output_dir:
        write_json(args.output_dir / 'results.json', payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
