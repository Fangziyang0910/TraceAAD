"""Check and recover stopped baseline searches from their retained journals.

Legacy journals do not contain RNG or stack state. Recovery restarts at a
method boundary, preserves spent budget, and explicitly records this difference.
Checking is read-only and never calls the model or runs task evaluations.
"""
import argparse
import copy
import importlib
import inspect
import json
import os
import fcntl
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from functools import lru_cache

from baselines.shared.observability import close_llm, shutdown_executor
from baselines.shared.sampling import SampleTrimmer
from baselines.shared.checkpoint import FIELDS, read_checkpoint, restore_checkpoint
from experiments.infra.runner import baseline_run_lock
from traceaad.storage import append_jsonl, read_json, rows, write_json
from experiments.infra.result_storage import Programs

METHODS = ('eoh', 'reevo', 'funsearch', 'mcts_ahd', 'calm', 'pathwise', 'shinka_evo')


@lru_cache(maxsize=1)
def profile_evaluator(task, workers):
    from experiments.infra.base import build_task
    from core import SecureEvaluator
    evaluation, _ = build_task(task, workers)
    return SecureEvaluator(evaluation)


def reprofile_saved(item):
    """Reprofile one program at a time, with the original per-instance workers."""
    key, code, score, task, workers = item
    outcome = profile_evaluator(task, workers).evaluate_program_with_details(code)
    return {'program_id': key, 'score': outcome.result, 'previous_score': score,
            'instance_scores': list(outcome.instance_scores), 'failure_kind': outcome.failure_kind}


def history(directory):
    directory = Path(directory)
    events = list(rows(directory / 'events.jsonl'))
    calls, seen = [], set()
    for path in (directory / 'calls.jsonl.gz', directory / 'calls.jsonl'):
        for call in rows(path):
            identity = call.get('transport_id')
            if identity and identity in seen:
                continue
            if identity:
                seen.add(identity)
            calls.append(call)
    state = read_json(directory / 'resume.json', {}).get('method_state', {})
    config = read_json(directory / 'run_config.json')
    if not config or config['method'] not in METHODS:
        raise ValueError('Not a supported baseline run')
    if config.get('status') == 'archived':
        raise ValueError('Archived experiments cannot be resumed')
    candidates = [r for r in events if r.get('kind') == 'candidate']
    checkpoint = read_checkpoint(directory) if config['method'] in FIELDS else None
    used = max([state.get('sample_count', state.get('sample_order', 0)),
                checkpoint['budget_used'] if checkpoint else 0]
               + [r.get('budget_used', 0) for r in candidates]
               + [r.get('data', {}).get('sample_order', 0) for r in events
                  if r.get('data', {}).get('counts_budget') is True])
    for call in calls:
        if call.get('error'):
            continue
        stage = call.get('stage')
        consumed = stage == 'generate' and config['method'] in ('eoh', 'reevo', 'funsearch')
        if config['method'] == 'shinka_evo':
            consumed = stage == 'patch'
        if config['method'] == 'mcts_ahd':
            consumed = stage == 'thought_alignment' or stage == 'generate' and (
                not call.get('function_parse_success') or call.get('parsed_thought') is None)
        if consumed:
            used = max(used, call.get('sample_order') or 0)
    if (not candidates and not checkpoint) or not 0 <= used <= config['budget'] or used == 0 and not checkpoint:
        raise ValueError('Missing history or inconsistent candidate budget')
    return config, events, calls, state, candidates, used


def build(config, directory):
    module = importlib.import_module('experiments.' + config['method'] + '.run')
    params = dict(config['method_params'])
    aliases = {'population_size': 'pop_size', 'selection_num': 'parents',
               'paper_generations': 'generations', 'reset_period_seconds': 'reset_period'}
    if config['method'] != 'eoh':
        aliases.pop('selection_num')
    params.update({aliases[k]: v for k, v in list(params.items()) if k in aliases})
    params.update(task=config['task'], backend=config['backend'],
                  seed=config['seed'], repeat=config['repeat'],
                  base_url=config['llm']['base_url'], model=config['llm']['model'],
                  no_proxy=config['llm']['no_proxy'], output_tokens=config['llm']['max_tokens'],
                  budget=config['budget'], max_sample_nums=config['budget'])
    accepted = inspect.signature(module.make_run_spec).parameters
    spec = module.make_run_spec(**{k: v for k, v in params.items() if k in accepted})
    # Validate constructor defaults against the original condition before writing logs.
    with tempfile.TemporaryDirectory() as temporary:
        module.write_run_config(spec, Path(temporary), config['run_name'])
        generated = read_json(Path(temporary) / 'run_config.json')
    for key in ('task_eval', 'method_params', 'llm'):
        a, b = dict(config[key]), dict(generated[key])
        if key == 'method_params':
            a.pop('scheduler_socket', None)
            b.pop('scheduler_socket', None)
        if a != b:
            raise ValueError(f'Original {key} differs from the current implementation')
    return module.build_method(spec, Path(directory) / 'logs')


def recover(method, directory, *, evaluate_profiles=False):
    config, events, calls, state, candidates, used = history(directory)
    directory = Path(directory)
    sources = Programs(directory)
    name = config['method']
    template = getattr(method, '_template_program', None)
    functions = {}
    for row in candidates:
        if name in FIELDS:
            continue
        if not row.get('valid'):
            continue
        code = sources.get(row['program']['id'])
        function = SampleTrimmer.sample_to_function(code, template)
        if function is None:
            raise ValueError(f"Cannot restore candidate {row['candidate_id']}")
        function.score = row['fitness']
        function.algorithm = row['attempt']['idea']
        function.operator = row['operator']
        function.sample_time = row['attempt']['sample_time']
        function.evaluate_time = row['attempt']['evaluate_time']
        functions[row['candidate_id']] = function
    if not functions and name not in FIELDS:
        raise ValueError('No valid historical programs')
    report = dict(method=config['method'], task=config['task'], budget_used=used,
                  budget=config['budget'], remaining=config['budget'] - used,
                  recovery='journal_boundary', rng='reseeded_not_restored',
                  valid_programs=sum(r.get('valid', False) for r in candidates), profiles_required=False)
    method._tot_sample_nums = used
    method._resume_mode = True
    profiler = method._profiler
    profiler._num_samples = max((r['candidate_id'] for r in candidates), default=0)
    profiler._budget_used = used
    profiler._resume_budget_from_method = True
    profiler._evaluate_success_program_num = sum(r.get('valid', False) for r in candidates)
    profiler._evaluate_failed_program_num = len(candidates) - profiler._evaluate_success_program_num
    profiler._canonical_best = copy.deepcopy(min(
        (r['program'] for r in candidates if r.get('valid')), key=lambda r: r['fitness'], default=None))
    profiler._token_usage = read_json(directory / 'token_usage.json', profiler._token_usage)
    profiler._llm_call_count = profiler._token_usage['calls']
    profiler._recorded_transport = {r['transport_id'] for r in calls if r.get('transport_id')}
    profiler._method_event_count = sum(r.get('kind') == 'method' for r in events)
    profiler._error_count = sum(1 for _ in rows(directory / 'logs/errors.jsonl'))
    profiler._resume_elapsed = max((r.get('progress', {}).get('elapsed', 0) for r in candidates), default=0)
    if used >= config['budget']:
        if config['method'] == 'funsearch' and (directory / 'recovery_profiles.jsonl').exists():
            restore_profile_ranking(method, candidates, functions, saved_profiles(directory, events, candidates))
            report['recovery'] = 'reevaluated_database'
        report['boundary'] = 'final_selection_only'
        return report

    if name in FIELDS:
        checkpoint = restore_checkpoint(method, directory, name)
        if not 0 <= checkpoint['budget_used'] <= used or checkpoint['candidate_count'] > profiler._num_samples:
            raise ValueError('Checkpoint budget or candidate count exceeds the retained history')
        method._resume_boundary = checkpoint['boundary']
        pending = used - checkpoint['budget_used']
        if name == 'pathwise' and pending:
            if method._initialization_complete:
                method._outer_iteration += 1  # Pending node IDs remain in the journal; do not reuse them.
            else:
                method._initialization_index += pending
        if name == 'pathwise':
            profiler._cur_gen = method._population.generation
        report.update(recovery='search_checkpoint', boundary=checkpoint['boundary'],
                      checkpoint_budget=checkpoint['budget_used'], pending_budget=pending,
                      rng='restored_at_checkpoint')
        return report

    def population():
        snapshots = [(i, r) for i, r in enumerate(events) if r.get('kind') == 'population']
        if not snapshots:
            raise ValueError('No recorded population')
        index, snapshot = snapshots[-1]
        members = []
        for member in snapshot['members']:
            member_source = member['function']
            function = SampleTrimmer.sample_to_function(member_source, template)
            if function is None:
                raise ValueError('Cannot parse saved population source')
            function.score, function.algorithm = member['score'], member['algorithm']
            members.append(function)
        method._population._population = members
        method._population._generation = snapshot['generation']
        profiler._cur_gen = snapshot['generation']
        return index

    if name == 'eoh':
        index = population()
        for row in events[index + 1:]:
            if row.get('kind') == 'candidate' and row['candidate_id'] in functions:
                method._population.register_function(functions[row['candidate_id']], increment_generation=False)
        method._population._generation = state.get('generation', method._population.generation)
        report['boundary'] = 'next_operator'
    elif name == 'reevo':
        population()
        method._elite_function = copy.deepcopy(min(functions.values(), key=lambda f: f.score))
        seeds = [functions[r['candidate_id']] for r in candidates
                 if r['operator'] == 'seed' and r['candidate_id'] in functions]
        if len(seeds) != 1:
            raise ValueError('Missing unique ReEvo seed')
        method._seed_function = seeds[0]
        reflections = [r for r in calls if r.get('stage') == 'long_term_reflection' and not r.get('error')]
        if reflections:
            method._long_term_reflection_str = reflections[-1]['response']
        report['boundary'] = 'new_generation_from_last_population_and_current_elite'
    elif name == 'funsearch':
        report['profiles_required'] = recover_database(method, directory, events, state, candidates, functions,
                                                       evaluate_profiles=evaluate_profiles)
        report['boundary'] = 'new_prompt_batch'
        if not report['profiles_required']:
            report['recovery'] = ('reevaluated_database' if (directory / 'recovery_profiles.jsonl').exists()
                                  else 'recorded_database')
            profiles = method._profiler._recovery_profiles
            report['rebuilt_programs'] = sum(p['score'] is not None for p in profiles.values())
            report['excluded_programs'] = sum(p['score'] is None for p in profiles.values())
        preparation = read_json(directory / 'recovery_ready.json', {})
        if report['profiles_required'] and preparation.get('status') == 'blocked':
            report['preparation_error'] = preparation['error']
    else:
        index = population()
        tree, accepted = recover_tree(method, events, candidates, functions, calls=calls)
        for row in events[index + 1:]:
            if row.get('kind') == 'candidate' and row['candidate_id'] in accepted:
                method._population.register_function(functions[row['candidate_id']])
        nodes = method.population_management([node.individual for node in accepted.values()], method._pop_size)
        method._recovered_tree = tree, nodes
        report.update(boundary='new_tree_iteration', tree_nodes=len(accepted))
    return report


def recover_tree(method, events, candidates, functions, *, calls=()):
    from baselines.mcts_ahd.mcts import MCTS, MCTSNode
    tree = MCTS('Root', method.alpha, method.lambda_0)
    accepted = {}
    first_expand = next((i for i, r in enumerate(events) if r.get('data', {}).get('event') == 'expand'), len(events))
    # Initialization registers root children but does not emit expand events.
    for row in events[:first_expand]:
        if row.get('kind') == 'candidate' and row['candidate_id'] in functions:
            function = functions[row['candidate_id']]
            if any(n.raw_info.score == function.score or n.code == str(function) for n in tree.root.children):
                continue
            if len(tree.root.children) >= method._init_pop_size:
                break
            accepted[row['candidate_id']] = method._add_root_child(tree, function)
    for row in events[first_expand:]:
        data = row.get('data', {})
        if data.get('event') != 'expand' or data.get('status') != 'expanded':
            continue
        identity = row.get('profiler_sample_order')
        function = functions.get(identity)
        if function is None or function.score != data['child_score']:
            raise ValueError('MCTS expansion cannot be matched to its candidate')
        parents = [tree.root] if data['root_parent'] else [n for n in accepted.values()
            if n.raw_info.score == data['parent_score'] and n.depth == data['parent_depth']
            and n.visits == data['parent_visits'] - 1
            and len(n.children) == data['parent_children_count'] - 1]
        if len(parents) > 1:
            prompts = [r['prompt'] for r in calls if r.get('stage') == 'generate'
                       and not r.get('error') and r.get('sample_order') == data['sample_order']]
            if len(prompts) == 1:
                parents = [n for n in parents if n.individual.body.strip() in prompts[0]]
        if len(parents) != 1:
            raise ValueError(f"MCTS parent identity is missing or ambiguous: candidate={identity}, "
                             f"score={data['parent_score']}, depth={data['parent_depth']}, "
                             f"visits={data['parent_visits']}, matches={len(parents)}")
        parent = parents[0]
        node = MCTSNode(function.algorithm, str(function), function.score,
                        individual=function, raw_info=function, parent=parent,
                        depth=parent.depth + 1, visit=1, Q=-function.score)
        if data['operator'] == 'e1':
            node.subtree.append(node)
        parent.add_child(node)
        tree.backpropagate(node)
        accepted[identity] = node
    if len(tree.root.children) < method._selection_num:
        raise ValueError('MCTS initialization is incomplete')
    return tree, accepted


def saved_profiles(directory, events, candidates):
    profiles = {r['program_id']: r for r in rows(directory / 'recovery_profiles.jsonl')}
    candidate_index = {r['candidate_id']: r for r in candidates}
    for row in events:
        data = row.get('data', {})
        candidate = candidate_index.get(row.get('profiler_sample_order'))
        if data.get('event') == 'evaluation_profile' and data.get('instance_scores') and candidate and candidate.get('valid'):
            profiles[candidate['program']['id']] = {'score': candidate['fitness'],
                'instance_scores': [-v for v in data['instance_scores']]}
    return profiles


def restore_profile_ranking(method, candidates, functions, profiles):
    method._profiler._recovery_profiles = profiles
    method._profiler._recovery_candidate_cutoff = max(r['candidate_id'] for r in candidates)
    rescored = [{**r['program'], 'fitness': profiles[r['program']['id']]['score'],
                 'score': profiles[r['program']['id']]['score']}
                for r in candidates if r['candidate_id'] in functions and profiles[r['program']['id']]['score'] is not None]
    method._profiler._canonical_best = min(rescored, key=lambda r: r['fitness']) if rescored else None


def recover_database(method, directory, events, state, candidates, functions, *, evaluate_profiles):
    db = method._database
    if state.get('island_resets') != 0:
        raise ValueError('Legacy island reset founders were not recorded; cannot reconstruct them')
    profiles = saved_profiles(directory, events, candidates)
    candidate_index = {r['candidate_id']: r for r in candidates}
    required = {r['program']['id'] for r in candidates if r['candidate_id'] in functions}
    use_profiles = evaluate_profiles or required <= profiles.keys()
    db._clock = lambda: 0.0  # Replay excludes downtime and cannot trigger island resets.
    db._last_reset_time = 0.0

    if evaluate_profiles:
        missing = {r['program']['id']: functions[r['candidate_id']] for r in candidates
                   if r['candidate_id'] in functions and r['program']['id'] not in profiles}
        sources = Programs(directory)
        config = read_json(directory / 'run_config.json')
        tasks = [(key, sources.get(key), function.score, config['task'], config['method_params']['eval_workers'])
                 for key, function in missing.items()]
        for index, item in enumerate(tasks, 1):
            record = reprofile_saved(item)
            profiles[record['program_id']] = record
            append_jsonl(directory / 'recovery_profiles.jsonl', record)
            if index % 10 == 0:
                print(json.dumps({'reprofile_completed': index, 'reprofile_total': len(missing)}), flush=True)

    def register(identity, island):
        function = functions.get(identity)
        if function is None:
            return
        row = candidate_index[identity]
        key = row['program']['id']
        if use_profiles:
            if profiles[key]['score'] is None:
                return
            vector = profiles[key]['instance_scores']
            if not vector or abs(sum(vector) / len(vector) - profiles[key]['score']) > 1e-9:
                raise ValueError('Cached instance scores do not match the new fitness')
            function.score = profiles[key]['score']
            function.instance_scores = tuple(-v for v in profiles[key]['instance_scores'])
        db.register_program(function, island, -function.score)

    seeds = [r for r in candidates if r['operator'] == 'seed']
    if len(seeds) != 1:
        raise ValueError('Missing unique FunSearch seed')
    register(seeds[0]['candidate_id'], None)
    for row in events:
        data = row.get('data', {})
        if data.get('event') == 'database_register' and data.get('status') == 'registered':
            register(row['profiler_sample_order'], data['island_id'])
    counts = [i.num_programs for i in db.islands]
    if not use_profiles and counts != state['island_program_counts']:
        raise ValueError('Recovered island membership/partition differs from recorded counts')
    if not use_profiles and [-v for v in db.best_score_per_island] != state['island_best_scores']:
        raise ValueError('Recovered island best scores differ')
    if use_profiles:
        if any(not island.clusters for island in db.islands):
            raise ValueError('Reevaluation left an island empty; cannot sample from it')
        restore_profile_ranking(method, candidates, functions, profiles)
    db._clock = time.time
    elapsed = max(r.get('progress', {}).get('elapsed', 0) for r in candidates)
    db._last_reset_time = time.time() - elapsed
    return not use_profiles


def close_recovery_method(method):
    shutdown_executor(getattr(method, '_evaluation_executor', None))
    providers = getattr(method, '_all_llms', None) or getattr(method, '_llms', None)
    if callable(providers):
        llms = providers()
    else:
        llms = [getattr(method, '_llm', None), getattr(getattr(method, '_sampler', None), 'llm', None)]
    seen = set()
    for llm in llms:
        if id(llm) not in seen:
            close_llm(llm)
            seen.add(id(llm))
    embedding = getattr(method, '_embedding_fn', None)
    close_llm(embedding)


def resume_manifest(path, *, execute=False, prepare_profiles=False):
    """Use the batch's existing sessions and logs, on its execution machine."""
    path = path.resolve()
    manifest = read_json(path)
    if not manifest or manifest.get('method') not in METHODS or manifest.get('status') == 'archived':
        raise ValueError('Not a resumable baseline batch')
    failed = False
    for plan in manifest['plan']:
        directory = path.parent / plan['task'] / plan['run_name']
        try:
            config, *_ = history(directory)
            if prepare_profiles and config['method'] != 'funsearch':
                raise ValueError('--prepare-profiles is only for FunSearch')
            if (config['method'], config['task'], config['run_name']) != (
                    manifest['method'], plan['task'], plan['run_name']):
                raise ValueError('Run identity differs from the manifest')
            summary = read_json(directory / 'summary.json', {})
            if summary.get('status') == 'finished' and (directory / 'selection.json').exists():
                print(f"{plan['run_name']}: finalized, skipped", flush=True)
                continue
            command = [sys.executable, '-u', '-m', 'experiments.infra.resume_baseline',
                       '--run-dir', str(directory)]
            if not execute and not prepare_profiles:
                subprocess.run(command, check=True)
                continue
            command.append('--execute' if execute else '--prepare-profiles')
            session = plan['session'] if execute else 'recover_profiles_' + plan['session']
            if subprocess.run(['tmux', 'has-session', '-t', session], capture_output=True).returncode == 0:
                print(f"{plan['run_name']}: session exists, skipped", flush=True)
                continue
            log = directory / 'logs' / ('resume_search.log' if execute else 'recovery_profiles.log')
            log.parent.mkdir(exist_ok=True)
            root = Path(__file__).resolve().parents[2]
            shell = f'cd {shlex.quote(str(root))} && exec {shlex.join(command)} >> {shlex.quote(str(log))} 2>&1'
            subprocess.run(['tmux', 'new-session', '-d', '-s', session, shell], check=True)
            print(f"{plan['run_name']}: launched; log={log}", flush=True)
        except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
            failed = True
            print(f"{plan['run_name']}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    if failed:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run-dir', type=Path)
    source.add_argument('--manifest', type=Path, help='check or resume all runs in a batch; execute in tmux')
    parser.add_argument('--execute', action='store_true', help='resume model calls; default only checks recovery')
    parser.add_argument('--prepare-profiles', action='store_true', help='CPU-only recovery of missing FunSearch instance scores')
    args = parser.parse_args()
    if args.execute and args.prepare_profiles:
        parser.error('Choose preparation or execution')
    if args.manifest:
        resume_manifest(args.manifest, execute=args.execute, prepare_profiles=args.prepare_profiles)
        return
    directory = args.run_dir.resolve()
    config, *_ = history(directory)
    if args.prepare_profiles and config['method'] != 'funsearch':
        parser.error('--prepare-profiles is only for FunSearch')
    from benchmarks.tasks import validate_task_interface
    validate_task_interface(config['task'], config)
    if args.execute or args.prepare_profiles:
        with baseline_run_lock(directory):
            from core.scheduling import scheduler_status
            socket = config['method_params'].get('scheduler_socket')
            if socket:
                scheduler_status(socket)
                os.environ['TRACEAAD_SCHEDULER_SOCKET'] = socket
            # Reprofiling is recovery work, not part of the original candidate evaluations.
            os.environ['TRACEAAD_EVALUATION_LOG'] = str(directory / 'recovery_evaluations.jsonl')
            if args.prepare_profiles:
                write_json(directory / 'recovery_ready.json', {'status': 'preparing', 'recovery': 'reevaluated_database'})
                try:
                    with tempfile.TemporaryDirectory() as temporary:
                        method = build(config, temporary)
                        try:
                            report = recover(method, directory, evaluate_profiles=True)
                        finally:
                            close_recovery_method(method)
                except Exception as exc:
                    write_json(directory / 'recovery_ready.json', {'status': 'blocked', 'error': str(exc)})
                    raise
                report['status'] = 'ready'
                write_json(directory / 'recovery_ready.json', report)
                print(json.dumps(report, ensure_ascii=False))
                return
            if read_json(directory / 'summary.json', {}).get('status') == 'finished' and (directory / 'selection.json').exists():
                raise ValueError('Run is already finalized; do not repeat final selection')
            method = build(config, directory)
            try:
                report = recover(method, directory, evaluate_profiles=True)
                report['resumed_at'] = datetime.now(timezone.utc).isoformat()
                append_jsonl(directory / 'recovery.jsonl', report)
                for path in directory.parent.parent.glob('batch_*.json'):
                    with (path.parent / '.batch.lock').open('a') as lock:
                        fcntl.flock(lock, fcntl.LOCK_EX)
                        manifest = read_json(path)
                        matching = [p for p in manifest.get('plan', []) if p.get('run_name') == config['run_name']]
                        if matching:
                            matching[0]['status'] = 'resuming'
                            matching[0]['recovery'] = dict(report)
                            manifest['status'] = 'partially_resumed'
                            write_json(path, manifest)
                summary = read_json(directory / 'summary.json', {})
                write_json(directory / 'summary.json', {**summary, 'status': 'running', 'writer_lock': True, 'recovery': report})
                if report['remaining'] == 0:
                    method._profiler.write_run_summary(status='finished', recovery=report)
                else:
                    os.environ['TRACEAAD_EVALUATION_LOG'] = str(directory / 'evaluations.jsonl')
                    method._profiler._evaluation._instance_execution['record_path'] = str(directory / 'evaluations.jsonl')
                    method.run()
            finally:
                close_recovery_method(method)
    else:
        with tempfile.TemporaryDirectory() as temporary:
            method = build(config, temporary)
            try:
                report = recover(method, directory)
            finally:
                close_recovery_method(method)
        print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
