"""Restore raw objectives and re-evaluate frozen programs on the CPU."""

import argparse
import copy
import json
import math
from pathlib import Path
import statistics

from benchmarks.tasks import CLASSES, FIXED_TASKS, FUNCTION_LIMITS, PRIMARY_SPLITS, heldout_task, training_task
from traceaad.common.config import REVISION
from traceaad.common.storage import read_json, write_json, save_heldout
from traceaad.common.instance_evaluation import InstanceProgramEvaluator
from experiments.infra.artifacts import pick_best_sample


def restore_record(record, rows):
    result = copy.deepcopy(record)
    if result.get('score_metric') == 'raw_objective':
        return result
    instances = result.get('instances', [])
    if result.get('valid') and (len(instances) != len(rows)
            or sorted(r['instance_index'] for r in instances) != list(range(len(rows)))):
        raise ValueError('raw objective requires every instance, not an aggregate deviation')
    if result.get('valid') and not math.isclose(statistics.fmean(r['score'] for r in instances),
                                               result['score'], abs_tol=1e-8):
        raise ValueError('record score does not match its instance vector')
    for item in instances:
        if item.get('valid'):
            reference = rows[item['instance_index']]['reference']
            value = reference * (1 + item['score'] / 100)
            if not math.isfinite(value) or value <= 0:
                raise ValueError('invalid reconstructed objective')
            item['score'] = round(value) if abs(value-round(value)) < 1e-8 else value
    result['score'] = statistics.fmean(r['score'] for r in instances) if result.get('valid') else None
    result['score_metric'] = 'raw_objective'
    return result


def convert_final_results(directory):
    directory = Path(directory)
    config = read_json(directory/'run_config.json')
    if config['task'] not in FIXED_TASKS:
        return None
    selection = read_json(directory/'selection.json')
    if not selection:
        return None
    if selection.get('score_metric') == 'raw_objective':
        return read_json(directory/'raw_objectives.json')
    task = CLASSES[config['task']](split='train', safe_evaluate=False)
    freeze = []
    for line in (directory/'events.jsonl').read_text().splitlines():
        event = json.loads(line)
        if event.get('progress', {}).get('phase') == 'freeze' and event.get('evaluations'):
            freeze.append(event['evaluations'])
    results = selection['results']
    if len(freeze) < len(results):
        raise ValueError(f'{directory}: missing final training instance records')
    vectors = freeze[-len(results):]
    updated = copy.deepcopy(selection)
    for old, new, records in zip(results, updated['results'], vectors):
        valid = [r for r in records if r['valid']]
        if old['fitness'] is not None:
            if not valid or not math.isclose(statistics.fmean(r['score'] for r in valid), old['fitness'], abs_tol=1e-8):
                raise ValueError('final training records are not aligned')
            new['fitness'] = statistics.fmean(restore_record(r, task._rows)['score'] for r in valid)
    updated['score_metric'] = 'raw_objective'
    selected = next(r for r in updated['results'] if r['node_id'] == updated['selected_node'])
    heldout = read_json(directory/'heldout.json', [])
    updated_heldout = copy.deepcopy(heldout)
    tests = {}
    for result in updated_heldout:
        evaluation = CLASSES[config['task']](split=result['split'], safe_evaluate=False)
        result['evaluations'] = [restore_record(r, evaluation._rows) for r in result['evaluations']]
        valid = [r for r in result['evaluations'] if r['valid']]
        if result['fitness'] is not None:
            result['fitness'] = result['objective'] = statistics.fmean(r['score'] for r in valid)
            result['scores'] = [r['score'] for r in valid]
        result['score_metric'] = 'raw_objective'
        result['scoring_revision'] = REVISION
        tests[result['split']] = result['fitness']
    payload = {'score_metric': 'raw_objective', 'task': config['task'],
               'selected_node': selection['selected_node'], 'selected_key': selection['selected_key'],
               'training': selected['fitness'], 'tests': tests}
    summary = read_json(directory/'summary.json')
    backup = directory/'logs/relative_objective_originals'
    backup.mkdir(parents=True, exist_ok=True)
    for name in ('selection.json', 'heldout.json', 'summary.json'):
        source = directory/name
        if source.exists() and not (backup/name).exists():
            (backup/name).write_bytes(source.read_bytes())
    summary['score_metric'] = 'raw_objective'
    summary['best']['fitness'] = summary['best']['score'] = selected['fitness']
    summary['selection_results'] = updated['results']
    write_json(directory/'raw_objectives.json', payload)
    write_json(directory/'heldout.json', updated_heldout)
    write_json(directory/'summary.json', summary)
    write_json(directory/'selection.json', updated)
    return payload


def reevaluate_final(directory, workers=8):
    directory = Path(directory)
    config = read_json(directory/'run_config.json')
    chosen, _ = pick_best_sample(directory)
    cached = read_json(directory/'raw_objectives.json', {})
    if (cached.get('evaluation_revision') == REVISION and cached.get('selected_key') == chosen['key']
            and cached.get('training') is not None and cached.get('tests')
            and all(v is not None for v in cached['tests'].values())):
        return cached
    if type(workers) is not int or workers < 1:
        raise ValueError('workers must be a positive integer')
    task = config['task']
    code, key = chosen['program'], chosen['key']
    options = dict(seeds=config['method_params'].get('evaluation_seeds', [730241]),
                   function_seconds=FUNCTION_LIMITS[task], function_budget_scope='mean_instance',
                   timeout_seconds=120, n_workers=workers, scheduler_socket=None)
    train, _ = training_task(task)
    evaluator = InstanceProgramEvaluator(train, role='selection', **options)
    training = evaluator.evaluate(code, key)
    results = []
    for split in PRIMARY_SPLITS[task]:
        evaluator = InstanceProgramEvaluator(heldout_task(task, split), role='heldout', **options)
        outcome = evaluator.evaluate(code, key)
        records = outcome['evaluations']
        result = {'task': task, 'split': split, 'scale': str(CLASSES[task].DATASET.SCALE)
                  if task in FIXED_TASKS else '50',
                  'key': key, 'node_id': chosen['node_id'], 'verification': 'verified',
                  'revision': REVISION, 'score_metric': 'raw_objective', 'protocol': evaluator.protocol,
                  'condition': 'shared', 'evaluation_seeds': list(options['seeds']),
                  'timeout_seconds': 120, 'function_seconds': FUNCTION_LIMITS[task],
                  'function_budget_scope': 'mean_instance', 'timeout_scope': 'instance', 'workers': workers,
                  'fitness': outcome['fitness'], 'objective': outcome['fitness'], 'evaluations': records,
                  'seconds': sum(r['seconds'] for r in records),
                  'scores': [r['score'] for r in records if r['valid']],
                  'failures': [r for r in records if not r['valid']],
                  'search_status': 'finished', 'sample_order': chosen['sample_order'], 'variant': ''}
        results.append(result)
    backup = directory/'logs/cpu_recompute_originals'
    backup.mkdir(parents=True, exist_ok=True)
    for name in ('heldout.json', 'summary.json', 'selection.json'):
        if not (backup/name).exists():
            (backup/name).write_bytes((directory/name).read_bytes())
    for result in results:
        save_heldout(directory, result)
    summary = read_json(directory/'summary.json')
    summary['score_metric'] = 'raw_objective'
    summary['best']['fitness'] = summary['best']['score'] = training['fitness']
    summary['final_training_evaluation'] = training
    selection = read_json(directory/'selection.json')
    selection['score_metric'] = 'raw_objective'
    for result in selection['results']:
        if result['node_id'] == chosen['node_id']:
            result['fitness'] = training['fitness']
    summary['selection_results'] = selection['results']
    write_json(directory/'summary.json', summary)
    write_json(directory/'selection.json', selection)
    payload = {'score_metric': 'raw_objective', 'task': task, 'selected_node': chosen['node_id'],
               'selected_key': key, 'training': training['fitness'],
               'tests': {r['split']: r['fitness'] for r in results},
               'evaluation_revision': REVISION, 'training_evaluation': training,
               'heldout': results}
    write_json(directory/'raw_objectives.json', payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dirs', nargs='+', type=Path)
    parser.add_argument('--evaluate', action='store_true', help='CPU evaluation of the frozen program')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    for directory in args.run_dirs:
        result = reevaluate_final(directory, args.workers) if args.evaluate else convert_final_results(directory)
        print(json.dumps({'run_dir': str(directory), 'result': None if result is None else
                         {k: result[k] for k in ('task', 'training', 'tests')}}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
