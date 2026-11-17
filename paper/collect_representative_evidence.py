"""Analyze fixed prefixes of the representative-cover runs; never run search.

The first invocation freezes complete JSONL records. Later invocations verify
and reuse those prefixes even if the original runs have continued.
"""
from pathlib import Path
from collections import Counter
import datetime
import hashlib
import json
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper/evidence'
DEST = OUT / 'representative_snapshot.json'
BATCH = '20261116_165500_evotrace_v3_rep_cover_co4'
MARKER = '\n\n[Earlier Algorithm-Design Experience]\n'


def normalized(s):
    return re.sub(r'\s+', ' ', s).strip()


def read_prefix(path, frozen=None):
    lines = path.read_bytes().splitlines(keepends=True)
    if lines and not lines[-1].endswith(b'\n'):
        lines.pop()
    if frozen:
        lines = lines[:frozen['attempts']]
    digest = hashlib.sha256(b''.join(lines)).hexdigest()
    if frozen:
        assert digest == frozen['sha256'], path
    return [json.loads(line) for line in lines], digest


def main():
    prior = json.loads(DEST.read_text()) if DEST.exists() else None
    by_source = {r['source']: r for r in prior['runs']} if prior else {}
    runs, case_candidates = [], []
    sources = sorted((ROOT / 'experiments_result/evotrace_v3_rep_cover').glob('*/' + BATCH + '*/native/attempts.jsonl'))
    for path in sources:
        source = str(path.relative_to(ROOT))
        rows, digest = read_prefix(path, by_source.get(source))
        cfg = json.loads((path.parent.parent / 'run_config.json').read_text())
        native = json.loads((path.parent / 'run_config.json').read_text())
        assert native['topology'] == {'max_representatives': 24, 'cover_distance': .15, 'region_size': 4}
        programs = {r['program']['id']: r['program'] for r in rows if r.get('program')}
        size = attached = elapsed = 0
        evs, tree_series, refs, regions, outputs = [], [], [], [], []
        updates = 0
        for row in rows:
            evaluations = row['attempt'].get('evaluations', [])
            evs.extend(evaluations)
            elapsed += len(evaluations)
            if evaluations and row.get('program'):
                size += 1
                attached += 1
            change = row['attempt'].get('record', {}).get('simplification')
            if change:
                updates += 1
                expected = set(change['triggers'][0]['removed_ids'])
                regional = []
                for kind, field in [('edge', 'path_contractions'), ('node', 'subtree_prunings')]:
                    for region in change[field]:
                        ids = region['removed_ids'] + region.get('branch_ids', [])
                        assert len(ids) >= 4
                        regional.extend(ids)
                        regions.append({'kind': kind, 'size': len(ids)})
                assert set(regional) == expected and len(regional) == len(expected)
                size -= len(expected)
                pending = []
                for call in change['reflection_calls']:
                    prompt = call['prompt']
                    block = normalized(prompt.split(MARKER, 1)[1].split('\n\n[Comparison Catalog]', 1)[0]) if MARKER in prompt else ''
                    matches = [o for o in outputs if o['normalized'] and o['normalized'] in block]
                    entry = {'kind': call['kind'], 'evaluation': elapsed,
                             'target_id': call['target_id'], 'prior_experience': bool(matches),
                             'input_tokens': call.get('usage', {}).get('prompt_tokens', 0) or 0,
                             'output_tokens': call.get('usage', {}).get('completion_tokens', 0) or 0,
                             'fallback': call['used_fallback']}
                    refs.append(entry)
                    region_key = 'path_contractions' if call['kind'] == 'edge' else 'subtree_prunings'
                    target_key = 'child_id' if call['kind'] == 'edge' else 'parent_id'
                    region = next(a for a in change[region_key] if a[target_key] == call['target_id'])
                    item = dict(entry, experience=call['experience'], normalized=normalized(call['experience']),
                                region=region, attempt_id=row['attempt']['id'], prompt=prompt)
                    if cfg['task'] == 'tsp_construct' and call['kind'] == 'edge':
                        for old in matches:
                            if old['kind'] == 'edge' and old['target_id'] in expected:
                                case_candidates.append({'source': source, 'seed': cfg['seed'],
                                                        'earlier': old, 'later': item,
                                                        'programs': programs})
                    pending.append(item)
                outputs.extend(pending)
            if 'tree_after' in row:
                assert size == len(row['tree_after']['nodes'])
            if evaluations:
                tree_series.append({'evaluation': elapsed, 'nodes': size})
        best, series = None, []
        for i, ev in enumerate(evs, 1):
            f = ev['evaluation'].get('fitness')
            if f is not None:
                best = f if best is None else min(f, best)
            series.append({'evaluation': i, 'fitness': f, 'best': best, 'status': ev['status']})
        config = {k: cfg[k] for k in ['task', 'seed', 'method_params', 'task_eval', 'evaluation_execution']}
        config['llm'] = {k: cfg['llm'][k] for k in ['model', 'temperature', 'max_tokens', 'enable_thinking', 'sampling']}
        config['native'] = {k: native[k] for k in ['topology', 'allocation', 'context', 'reflection_temperature', 'reflection_max_tokens']}
        runs.append({'task': cfg['task'], 'seed': cfg['seed'], 'source': source, 'sha256': digest,
                     'attempts': len(rows), 'evaluations': len(evs), 'attached_nodes': attached,
                     'retained_nodes': size, 'removed_nodes': attached-size, 'updates': updates,
                     'reflection_calls': refs, 'regions': regions, 'series': series,
                     'tree_series': tree_series, 'config': config})
    assert len(runs) == 12
    horizon = min(r['evaluations'] for r in runs) // 10 * 10
    aggregates = []
    for task in ['tsp_construct', 'cvrp_aco', 'fssp_gls', 'op_aco']:
        group = sorted((r for r in runs if r['task'] == task), key=lambda r:r['seed'])
        scores = [r['series'][horizon-1]['best'] for r in group]
        aggregates.append({'task': task, 'scores': scores, 'mean': statistics.mean(scores), 'sd': statistics.stdev(scores)})
    allrefs = [c for r in runs for c in r['reflection_calls']]
    totals = {k: sum(r[k] for r in runs) for k in ['evaluations', 'attached_nodes', 'retained_nodes', 'removed_nodes', 'updates']}
    totals.update({'reflections': len(allrefs), 'kinds': dict(Counter(c['kind'] for c in allrefs)),
                   'with_prior_experience': sum(c['prior_experience'] for c in allrefs),
                   'inherited_by_kind': dict(Counter(c['kind'] for c in allrefs if c['prior_experience'])),
                   'input_tokens': sum(c['input_tokens'] for c in allrefs),
                   'output_tokens': sum(c['output_tokens'] for c in allrefs),
                   'fallbacks': sum(c['fallback'] for c in allrefs),
                   'region_sizes': dict(sorted(Counter(c['size'] for r in runs for c in r['regions']).items()))})
    result = {'captured_at': prior['captured_at'] if prior else datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'batch': BATCH, 'scope': 'Current representative-cover method; committed training prefixes only.',
              'common_budget': horizon, 'runs': runs, 'aggregates': aggregates, 'totals': totals}
    OUT.mkdir(exist_ok=True)
    DEST.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'common_budget': horizon, 'totals': totals, 'aggregates': aggregates}, indent=2))

    # Select by chronology and record availability, not by an improvement score.
    for candidate in sorted(case_candidates, key=lambda c:(c['later']['evaluation'], c['seed'], c['earlier']['evaluation'])):
        path = ROOT / candidate['source']
        calls, calls_hash = read_prefix(path.parent / 'calls.jsonl')
        later = candidate['later']
        matching = [c for c in calls if c['attempt_id'] > later['attempt_id']
                    and c['stage'] == 'generate'
                    and later['normalized'] in normalized(c['prompt'])]
        frozen_run = next(r for r in runs if r['source'] == candidate['source'])
        max_id = max(r['attempt']['id'] for r in read_prefix(path, frozen_run)[0])
        matching = [c for c in matching if c['attempt_id'] <= max_id]
        if not matching:
            continue
        follow = matching[0]
        ids = set()
        for phase in ['earlier', 'later']:
            part = candidate[phase]
            ids.update(part['region']['removed_ids'] + part['region'].get('branch_ids', []))
            ids.update([part['region']['parent_id'], part['region']['child_id']])
            part.pop('normalized')
        frozen_rows = read_prefix(path, frozen_run)[0]
        following_row = next(r for r in frozen_rows if r['attempt']['id'] == follow['attempt_id'])
        candidate['following'] = {'attempt_id': follow['attempt_id'], 'prompt': follow['prompt'],
                                  'response': follow['response'], 'program': following_row['program'],
                                  'context': following_row['attempt'].get('record',{}).get('context')}
        candidate['programs'] = [candidate['programs'][i] for i in sorted(ids) if i in candidate['programs']]
        evaluated_at = {r['program']['id']: r['attempt']['evaluations'][-1]['id']
                        for r in frozen_rows if r.get('program') and r['attempt'].get('evaluations')}
        for program in candidate['programs']:
            program['evaluation_index'] = evaluated_at[program['id']]
        candidate['following']['evaluation_index'] = evaluated_at[following_row['program']['id']]
        candidate['selection_rule'] = 'Earliest later reflection (ties: seed, earlier reflection) among TSP runs that incorporates an earlier edge note, removes its carrier, and supplies the resulting note to a subsequent recorded generation call.'
        candidate['source_prefix_sha256'] = frozen_run['sha256']
        candidate['calls_source'] = str((path.parent / 'calls.jsonl').relative_to(ROOT))
        # Freeze only the relevant call for portability; bind it to its source record.
        candidate['following_call_sha256'] = hashlib.sha256(json.dumps(follow, sort_keys=True).encode()).hexdigest()
        (OUT / 'inheritance_case.json').write_text(json.dumps(candidate, ensure_ascii=False, indent=2) + '\n')
        print('case', candidate['seed'], candidate['earlier']['evaluation'], candidate['later']['evaluation'], candidate['following']['attempt_id'])
        break
    else:
        raise RuntimeError('No complete inheritance-to-generation case in the frozen TSP prefixes.')


if __name__ == '__main__':
    main()
