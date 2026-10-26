"""Verify and summarize the frozen online-loop batch, including every call."""

import argparse
from collections import Counter
from pathlib import Path

from traceaad.storage import read_json, write_json
from .run import canonical, digest


def costs(directory):
    calls = [c for path in sorted(directory.glob('calls/*.json'))
             for c in read_json(path)['details']['calls']]
    return dict(calls=len(calls), measured_tokens=sum(c['usage'].get('total_tokens', 0) for c in calls),
                missing_usage=sum('total_tokens' not in c['usage'] for c in calls),
                transport_errors=sum(bool(c['error']) for c in calls),
                endpoints=dict(Counter((c.get('scheduler') or {}).get('endpoint', c.get('model')) for c in calls)))


def summarize_run(directory):
    summary, state = read_json(directory / 'summary.json'), read_json(directory / 'state.json')
    if summary is None:
        raise ValueError(f'incomplete: {directory}')
    config = read_json(directory / 'run_config.json')
    curve, events = state['curve'], state['events']
    programs = {p['id']: p for p in state['programs']}
    selected = read_json(directory / 'selection.json')
    budget = config['budget']
    cost = costs(directory)
    assert cost['calls'] == len(events) == len(curve) == budget
    assert [v['calls'] for v in curve] == list(range(1, budget + 1))
    assert [v['best'] for v in curve] == sorted((v['best'] for v in curve), reverse=True)
    best = min((p for p in programs.values() if p['status'] == 'valid'), key=lambda p: (p['fitness'], p['id']))
    assert best['id'] == selected['id'] == summary['winner'] == curve[-1]['selected']
    assert digest(best['code']) == selected['code_sha256']
    for p in programs.values():
        if p['status'] == 'pending':
            raise ValueError('unevaluated program')
    for path in directory.glob('evaluations/*.json'):
        data = read_json(path)
        matching = [p for p in programs.values() if digest(p['code']) == data['code_sha256']]
        assert matching, path
    ids = set()
    node = best
    while node:
        ids.add(node['id'])
        node = programs.get(node['parent_id'])
    applications = []
    for event in events:
        if not event.get('transformation_id'):
            continue
        operator = next(t for t in state['bank'] if t['id'] == event['transformation_id'])
        parent = programs[event['parent']]
        child = programs.get(event['child'])
        improved = child is not None and child['status'] == 'valid' and child['fitness'] < parent['fitness'] - 1e-9
        applications.append(dict(call=event['call'], operator=operator['id'], parent=parent['id'],
                                 child=child['id'] if child else None, improved=improved,
                                 different_parent=parent['id'] != operator['source_parent'],
                                 on_winner_path=child is not None and child['id'] in ids,
                                 unchanged=child is not None and canonical(child['code']) == canonical(parent['code'])))
    evaluations = [read_json(p) for p in directory.glob('evaluations/*.json')]
    generations = [e for e in events if e['kind'] == 'candidate']
    return dict(**summary, cost=cost, first_improvement_call=next((v['calls'] for v in curve
               if v['best'] < summary['initial_best'] - 1e-9), None),
               train_gain=(summary['initial_best'] - summary['train']) / summary['initial_best'],
               inductions=sum(e['kind'] == 'induce' for e in events),
               opportunities=sum(e.get('offered', False) for e in events),
               delivery_errors=sum(bool(e['error']) for e in generations),
               runtime_failures=sum(p['status'] != 'valid' for p in programs.values()),
               evaluation_count=len(evaluations),
               function_cpu_seconds=sum(e.get('function_cpu_seconds', 0) or 0 for e in evaluations),
               applications=applications)


def analyze(out):
    runs = {f'{arm}_{seed}': summarize_run(out / (f'qwen_{arm}_{seed}' + ('_corrected' if arm == 'bank' else '')))
            for seed in range(3) for arm in ('open', 'bank')}
    paired = []
    for seed in range(3):
        a, b = runs[f'open_{seed}'], runs[f'bank_{seed}']
        assert a['initial_best'] == b['initial_best']
        token_cap = min(a['tokens'], b['tokens'])
        token_prefix = {}
        for arm in ('open', 'bank'):
            directory = out / (f'qwen_{arm}_{seed}' + ('_corrected' if arm == 'bank' else ''))
            curve = read_json(directory / 'state.json')['curve']
            tokens, count, best = 0, 0, a['initial_best']
            for path in sorted(directory.glob('calls/*.json')):
                usage = read_json(path)['details']['calls'][0]['usage']
                if 'total_tokens' not in usage:
                    raise ValueError('cannot compare token prefixes with missing usage')
                if tokens + usage['total_tokens'] > token_cap:
                    break
                tokens += usage['total_tokens']
                count += 1
                best = curve[count - 1]['best']
            token_prefix[arm] = dict(calls=count, tokens=tokens, best_train=best)
        paired.append(dict(seed=seed, strength=('weak', 'middle', 'strong')[seed],
                           open_train=a['train'], bank_train=b['train'],
                           open_test=a['heldout'], bank_test=b['heldout'],
                           bank_relative_train_gain=(a['train']-b['train'])/a['train'],
                           bank_relative_test_gain=(a['heldout']-b['heldout'])/a['heldout']
                           if a['heldout'] is not None and b['heldout'] is not None else None,
                           posthoc_common_token_cap=token_cap, token_prefix=token_prefix))
    execution = {}
    for backend, name in (('qwen', 'qwen'), ('deepseek', 'deepseek_local')):
        directory = out / 'execution' / name
        data = read_json(directory / 'results.json')
        assert len(data) == 12
        execution[backend] = dict(attempts=12, valid=sum(r['valid'] for r in data),
                                  matched=sum(r['matched'] for r in data), cost=costs(directory),
                                  failures=[{k: v for k, v in r.items() if k not in ('code', 'reply')}
                                            for r in data if not r['matched']])
    for i in range(1, 13):
        q, d = [read_json(out / 'execution' / name / 'calls' / f'{i:03}.json') for name in ('qwen', 'deepseek_local')]
        assert q['prompt_sha256'] == d['prompt_sha256']
    infrastructure = costs(out / 'execution' / 'deepseek')
    unique_calls = {}
    for path in out.glob('**/calls/*.json'):
        for call in read_json(path)['details']['calls']:
            unique_calls[call['transport_id']] = call
    all_calls = dict(recorded_unique_requests=len(unique_calls),
                    measured_tokens=sum(c['usage'].get('total_tokens', 0) for c in unique_calls.values()),
                    missing_usage=sum('total_tokens' not in c['usage'] for c in unique_calls.values()),
                    interrupted_unrecorded_requests_upper_bound=5)
    for seed in range(3):
        for path in (out / f'qwen_bank_{seed}_corrected').glob('calls/*.json'):
            prompt = read_json(path)['details']['calls'][0]['prompt']
            assert 'function_seconds_limit' not in prompt
            assert 'mean function CPU time' not in prompt
    result = dict(runs=runs, paired=paired, execution=execution, infrastructure=infrastructure,
                  whole_batch_cost=all_calls, correction=read_json(out / 'feedback_correction.json'))
    write_json(out / 'analysis.json', result)
    print(paired)
    print({k: {f: v[f] for f in ('attempts', 'valid', 'matched')} for k, v in execution.items()})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    analyze(parser.parse_args().out)
