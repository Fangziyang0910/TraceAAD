"""Evaluate committed programs with the same task and execution path as search.

    uv run python -m experiments.infra.evaluate RUN_DIR [...] --units 50,100,200

``shared`` keeps the baseline condition: no additional candidate RNG seed and
ACO seed 1234. Non-ACO tasks run their complete instance set serially. ``traceaad`` uses the run's evaluation_seeds. Results carry both
the condition and implementation revision; stored historical scores stay intact.
"""

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.tasks import TASKS, SPLITS, heldout_task, scale_of_split, split_of_scale
from traceaad.common.config import REVISION
from traceaad.common.evaluation import ProgramEvaluator
from traceaad.common.storage import heldout_identity, read_json, save_heldout, write_json
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


def evaluate_run(run_dir, splits, *, condition='shared', workers=4, timeout_seconds=None,
                 sample_order=None, max_sample_order=None, allow_incomplete=False,
                 variant=None):
    run_dir = Path(run_dir)
    config = read_json(run_dir / 'run_config.json')
    task = config['task']
    chosen, _ = pick_best_sample(run_dir, sample_order=sample_order,
        max_sample_order=max_sample_order, allow_incomplete=allow_incomplete)
    summary = load_run_summary(run_dir, require_finished=not allow_incomplete)
    code = chosen['program']
    key = hashlib.sha256(code.encode()).hexdigest()
    seeds = tuple(config['method_params']['evaluation_seeds']) if condition == 'traceaad' else (None,)
    variant = variant or f'{condition}:{REVISION}'
    results = []
    for split in splits:
        evaluation = heldout_task(task, split, workers, timeout_seconds)
        evaluator = ProgramEvaluator(evaluation, seeds, 'heldout', measure_calls=condition == 'traceaad')
        result = {'task': task, 'split': split, 'scale': str(scale_of_split(task, split)),
                  'key': key, 'node_id': chosen['node_id'], 'verification': 'verified'}
        result['verification'] = heldout_identity(run_dir, result)
        previous = next((row for row in read_json(run_dir / 'heldout.json', [])
                         if (row['variant'], row['scale']) == (variant, result['scale'])), None)
        if previous:
            if (previous['key'], previous.get('node_id'), previous.get('protocol')) != (key, chosen['node_id'], evaluator.protocol):
                raise ValueError('existing result belongs to another program or evaluation condition')
            results.append({**previous, "verification": result["verification"]})
            continue
        outcome = evaluator.evaluate(code, key)
        records = outcome['evaluations']
        score = outcome['fitness']
        result.update(condition=condition, revision=REVISION, protocol=evaluator.protocol,
                      evaluation_seeds=list(seeds), timeout_seconds=evaluation.timeout_seconds,
                      workers=getattr(evaluation, 'n_workers', 1), fitness=score,
                      objective=score,
                      seconds=sum(row['seconds'] for row in records), scores=[row['score'] for row in records if row['valid']],
                      failures=[{'kind': row['failure_kind'], 'error': row['error']} for row in records if not row['valid']],
                      search_status=summary['status'], sample_order=chosen['sample_order'], variant=variant)
        save_heldout(run_dir, result, variant)
        results.append(result)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dirs', nargs='+', type=Path)
    parser.add_argument('--task', choices=TASKS)
    parser.add_argument('--units')
    parser.add_argument('--timeout', type=float)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--condition', choices=('shared', 'traceaad'), default='shared')
    parser.add_argument('--variant', help='default: condition plus implementation revision')
    parser.add_argument('--sample-order', type=int)
    parser.add_argument('--max-sample-order', type=int)
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if args.sample_order is not None and len(args.run_dirs) != 1:
        raise ValueError('--sample-order requires one run directory')
    payload = []
    for run in args.run_dirs:
        task = read_json(run / 'run_config.json')['task']
        if args.task and task != args.task:
            raise ValueError(f'{run}: task is {task}, not {args.task}')
        results = evaluate_run(run, parse_units(task, args.units), condition=args.condition,
            workers=args.workers, timeout_seconds=args.timeout, sample_order=args.sample_order,
            max_sample_order=args.max_sample_order, allow_incomplete=args.allow_incomplete,
            variant=args.variant)
        payload.extend({'run_dir': str(run), **result} for result in results)
    if args.output_dir:
        write_json(args.output_dir / 'results.json', payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
