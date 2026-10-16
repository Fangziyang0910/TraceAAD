"""Convert finished relative-score archives; CPU-fill missing instance measurements."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil

from benchmarks.tasks import CLASSES, FIXED_TASKS
from experiments.infra.raw_objectives import convert_final_results, restore_record
from traceaad.v12_1.instance_evaluation import InstanceProgramEvaluator
from traceaad.v12_1.storage import Programs, append_jsonl, read_json, repair_jsonl_tail, rows, write_json


def signature(record):
    return hashlib.sha256(json.dumps([record['key'], record.get('protocol'),
                                     record['score']], separators=(',', ':')).encode()).hexdigest()


def exact_profile(data, references):
    # FunSearch's database maximizes the negative evaluation score.
    values = [-v for v in data['instance_scores']]
    return restore_record({'valid': True, 'score': data['score'], 'instances': [
        {'instance_index': i, 'valid': True, 'score': v} for i, v in enumerate(values)]}, references)


def plan(events, evaluations, references):
    by_key = defaultdict(list)
    for record in evaluations:
        if record.get('valid'):
            by_key[record['key']].append(record)
    candidates, jobs, exact = {}, {signature(r): r for r in evaluations if r.get('valid')}, {}
    for event in events:
        if event['kind'] == 'candidate' and event.get('valid'):
            p = event['program']
            matches = [r for r in by_key[p['key']] if r['score'] == event['fitness']]
            if not matches:
                raise ValueError(f"candidate {event['candidate_id']} lacks its evaluation conditions")
            # Identical key/score records with different limits need chronological alignment.
            record = matches[0]
            by_key[p['key']].remove(record)
            identity = signature(record)
            jobs[identity] = record
            candidates[event['candidate_id']] = identity
        data = event.get('data', {})
        if data.get('event') == 'evaluation_profile' and data.get('instance_scores'):
            identity = candidates.get(event.get('profiler_sample_order'))
            if identity is None:
                raise ValueError('instance profile has no matching candidate')
            record = jobs[identity]
            if not math.isclose(data['score'], record['score'], rel_tol=0, abs_tol=1e-8):
                raise ValueError('instance profile does not match candidate score')
            restored = exact_profile(data, references)
            exact[identity] = {'identity': identity, 'key': record['key'],
                               'source': 'exact_conversion', 'fitness': restored['score'],
                               'evaluations': [restored]}
    return candidates, jobs, exact


def execution(record, config):
    params = config['method_params']
    return dict(seeds=params.get('evaluation_seeds', [730241]),
                timeout_seconds=record['timeout_seconds'],
                function_seconds=record['function_seconds_limit'],
                function_budget_scope=record.get('function_budget_scope', 'instance'))


def write_rows(path, records):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def publish(directory, backup, events, candidates, results, references):
    by_pair = defaultdict(list)
    originals = list(rows(backup / 'evaluations.jsonl'))
    for record in originals:
        result = results.get(signature(record)) if record.get('valid') else None
        if result:
            by_pair[(record['key'], record['score'])].append(result)
    for event in events:
        if event.get('candidate_id') in candidates:
            key = event.get('attempt', {}).get('function_key')
            if key:
                by_pair[(key, event['fitness'])].append(results[candidates[event['candidate_id']]])

    def score_fields(obj, result):
        for field in ('score', 'fitness'):
            if field in obj:
                obj[field] = result['fitness']
        obj['score_metric'] = 'raw_objective'
        obj['score_source'] = result['source']
        obj['raw_measurement'] = result['identity']

    for event in events:
        if event['kind'] == 'candidate':
            identity = candidates.get(event['candidate_id'])
            if identity:
                score_fields(event, results[identity])
                score_fields(event['program'], results[identity])
            else:
                event['score_metric'] = 'raw_objective'
        elif event['kind'] == 'population':
            for member in event['members']:
                matches = by_pair.get((member['function_key'], member['score']), [])
                old = member.pop('score')
                member['historical_score'] = old
                member['score'] = matches[0]['fitness'] if matches else None
                member['score_metric'] = 'raw_objective'
        data = event.get('data', {})
        # These describe the decisions actually made, not a replay under a new metric.
        for field in ('score', 'parent_score', 'child_score', 'elite_score', 'instance_scores'):
            if field in data:
                data['historical_' + field] = data.pop(field)
        if data:
            data['historical_score_metric'] = 'relative_deviation'
        if event.get('evaluations'):
            event['evaluations'] = [restore_record(r, references) for r in event['evaluations']]
            event['score_metric'] = 'raw_objective'
    write_rows(directory / 'events.jsonl', events)
    for record in originals:
        if record.get('valid'):
            result = results.get(signature(record))
            if result:
                score_fields(record, result)
            else:
                record['score'] = None
                record['score_metric'] = 'raw_objective'
                record['score_source'] = 'missing_instance_measurement'
        else:
            record['score_metric'] = 'raw_objective'
    write_rows(directory / 'evaluations.jsonl', originals)
    # Recovery profiles are separate CPU measurements, never substituted for old candidates.
    profiles = []
    recovery_scores = {}
    for row in rows(backup / 'recovery_profiles.jsonl'):
        if 'previous_score' in row:
            row['historical_previous_score'] = row.pop('previous_score')
        if row['score'] is None:
            profiles.append({**row, 'score_metric': 'raw_objective'})
            continue
        restored = restore_record({'valid': True, 'score': row['score'], 'instances': [
            {'instance_index': i, 'score': v, 'valid': True}
            for i, v in enumerate(row['instance_scores'])]}, references)
        profiles.append({**row, 'score': restored['score'], 'score_metric': 'raw_objective',
                         'instance_scores': [r['score'] for r in restored['instances']]})
        recovery_scores[(row['key'], row['score'])] = restored['score']
    if profiles:
        write_rows(directory / 'recovery_profiles.jsonl', profiles)
    recovery = list(rows(backup / 'recovery_evaluations.jsonl'))
    for row in recovery:
        if row.get('valid'):
            row['score'] = recovery_scores.get((row['key'], row['score']))
        row['score_metric'] = 'raw_objective'
    if recovery:
        write_rows(directory / 'recovery_evaluations.jsonl', recovery)
    resume = read_json(directory / 'resume.json', {})
    if 'events.jsonl' in resume.get('files', {}):
        resume['files']['events.jsonl'] = (directory / 'events.jsonl').stat().st_size
    if resume.get('method_state'):
        resume['method_state']['score_metric'] = 'historical_relative_deviation'
    write_json(directory / 'resume.json', resume)
    (directory / '.cache/history.json').unlink(missing_ok=True)


def run(directory, workers, scheduler_socket, prepare_only=False):
    directory = Path(directory)
    with (directory / '.raw_objectives.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return convert(directory, workers, scheduler_socket, prepare_only)


def convert(directory, workers, scheduler_socket, prepare_only):
    status_file = directory / 'raw_search.json'
    previous = read_json(status_file, {})
    if previous.get('status') == 'complete':
        return previous
    config = read_json(directory / 'run_config.json')
    if config['task'] not in FIXED_TASKS:
        raise ValueError('only relative-score fixed tasks need conversion')
    if read_json(directory / 'summary.json')['status'] != 'finished':
        raise ValueError('conversion requires a finished search')
    backup = directory / 'logs/raw_search_originals'
    backup.mkdir(parents=True, exist_ok=True)
    for name in ('events.jsonl', 'evaluations.jsonl', 'selection.json', 'heldout.json',
                 'summary.json', 'resume.json', 'recovery_profiles.jsonl', 'recovery_evaluations.jsonl'):
        if (directory / name).exists() and not (backup / name).exists():
            shutil.copy2(directory / name, backup / name)
    task = CLASSES[config['task']](split='train', safe_evaluate=False)
    recorded = read_json(directory / 'exact_objective_conversion.json')['summary']['reference_values']
    if recorded != [row['reference'] for row in task._rows]:
        raise ValueError('historical reference instances differ')
    events = list(rows(backup / 'events.jsonl'))
    evaluations = list(rows(backup / 'evaluations.jsonl'))
    candidates, jobs, exact = plan(events, evaluations, task._rows)
    log = directory / 'raw_search_evaluations.jsonl'
    if log.exists():
        repair_jsonl_tail(log, backup)
    results = {r['identity']: r for r in rows(log)}
    for identity, result in exact.items():
        if identity not in results:
            append_jsonl(log, result)
            results[identity] = result
    convert_final_results(directory)

    def retry_timeout(result):
        return (result.get('failure') or {}).get('kind') == 'timeout' and result['source'] != 'cpu_reevaluation_relaxed'

    if previous.get('status') == 'complete_with_missing' and not any(retry_timeout(r) for r in results.values()):
        return previous

    def status(state):
        report = {'status': state, 'score_metric': 'raw_objective', 'run_dir': str(directory),
                  'updated_at': datetime.now(timezone.utc).isoformat(),
                  'valid_candidates': len(candidates), 'unique_measurements': len(jobs),
                  'exact_candidates': sum(i in exact for i in candidates.values()),
                  'processed': len(results), 'pending': len(set(jobs) - set(results)) + sum(retry_timeout(r) for r in results.values()),
                  'missing_candidates': sum(i in results and results[i]['fitness'] is None
                                            for i in candidates.values()),
                  'missing_measurements': sum(r['fitness'] is None for r in results.values()),
                  'relaxed_measurements': sum(r['source'] == 'cpu_reevaluation_relaxed' for r in results.values()),
                  'backup': 'logs/raw_search_originals',
                  'workers': workers, 'scheduler_socket': scheduler_socket}
        write_json(status_file, report)
        return report

    status('prepared' if prepare_only else 'running')
    if prepare_only:
        return read_json(status_file)
    programs = Programs(directory)
    for identity, record in jobs.items():
        previous_result = results.get(identity)
        relaxed = previous_result is not None and retry_timeout(previous_result)
        if previous_result is not None and not relaxed:
            continue
        code = programs.get(record['key'])
        if not code:
            result = {'identity': identity, 'key': record['key'], 'source': 'missing_program',
                      'fitness': None, 'failure': {'kind': 'missing_program'}, 'evaluations': []}
            append_jsonl(log, result)
            results[identity] = result
            status('running')
            continue
        if hashlib.sha256(code.encode()).hexdigest() != record['key']:
            raise ValueError('changed candidate program')
        options = execution(record, config)
        if relaxed:
            options.update(function_seconds=None, timeout_seconds=max(600, options['timeout_seconds']))
        evaluator = InstanceProgramEvaluator(task, role='raw_backfill', n_workers=workers,
                                             scheduler_socket=scheduler_socket, **options)
        outcome = evaluator.evaluate(code, record['key'])
        retry = previous_result if relaxed else None
        if outcome['fitness'] is None and not relaxed:
            retry = outcome
            if outcome['failure']['kind'] == 'timeout':
                relaxed = True
                options.update(function_seconds=None, timeout_seconds=max(600, options['timeout_seconds']))
                evaluator = InstanceProgramEvaluator(task, role='raw_backfill', n_workers=workers,
                                                     scheduler_socket=scheduler_socket, **options)
            outcome = evaluator.evaluate(code, record['key'])
        result = {'identity': identity, 'key': record['key'],
                  'source': 'cpu_reevaluation_relaxed' if relaxed else 'cpu_reevaluation',
                  'historical_protocol': record.get('protocol'),
                  'score_metric': 'raw_objective', 'fitness': outcome['fitness'],
                  'evaluations': outcome['evaluations'], 'failure': outcome['failure']}
        if retry:
            result['previous_attempt'] = retry
        append_jsonl(log, result)
        results[identity] = result
        report = status('running')
        print(json.dumps(report), flush=True)
    publish(directory, backup, copy.deepcopy(events), candidates, results, task._rows)
    report = status('complete_with_missing' if any(r['fitness'] is None for r in results.values()) else 'complete')
    summary = read_json(directory / 'summary.json')
    summary['raw_search'] = {k: report[k] for k in ('status', 'valid_candidates', 'exact_candidates', 'missing_candidates', 'relaxed_measurements')}
    write_json(directory / 'summary.json', summary)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dirs', nargs='+', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--parallel', type=int, default=3)
    parser.add_argument('--scheduler-socket', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.parallel < 1:
        parser.error('worker counts must be positive')
    with ProcessPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(run, p, args.workers, args.scheduler_socket, args.prepare_only)
                   for p in args.run_dirs]
        for future in futures:
            print(json.dumps(future.result()), flush=True)


if __name__ == '__main__':
    main()
