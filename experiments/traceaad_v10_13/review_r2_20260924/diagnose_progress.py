"""Offline diagnostics bounded by the saved 01:22 protocol audit; never evaluates code."""
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'experiments/traceaad_v10_13/results/runtime_20260924_v1013r2'))
from traceaad.v10_13.parsing import response_object, THINK_BLOCK_RE
from traceaad.v10_13.selection import code_key
from traceaad.v10_13.storage import read_journal

HERE = Path(__file__).resolve().parent
RESULTS = ROOT / 'experiments/traceaad_v10_13/results'
BOUND = json.loads((HERE / 'progress_0121_protocol.json').read_text())
LIMITS = {r['run_name']: r['effective_candidate_boundary'] for r in BOUND['runs']}


def load_batch(batch, limit=None):
    manifest = json.loads((RESULTS / f'batch_{batch}.json').read_text())
    result = {}
    for lane in manifest['plan']:
        d = RESULTS / lane['task'] / lane['run_name']
        state = json.loads((d / 'tree_state.json').read_text())
        boundary = limit[lane['run_name']] if limit else state['completed_candidates']
        ev = [e for e in read_journal(d / 'events.jsonl') if e['candidate_id'] <= boundary]
        ids = {e['node_id'] for e in ev if e.get('node_id') is not None}
        nodes = {n['id']: n for n in state['nodes'] if n['id'] in ids}
        calls = [c for c in read_journal(d / 'llm_calls.jsonl') if c['candidate_id'] <= boundary]
        result[lane['task'], lane['repeat']] = dict(lane=lane, state=state, events=ev, nodes=nodes, calls=calls)
    return result


def best_at(run, budget):
    return max(e['fitness'] for e in run['events'] if e.get('evaluation_id') and
               e['evaluation_id'] <= budget and e.get('fitness') is not None)


def prefix_cost(run, budget):
    end = next(e for e in run['events'] if e.get('evaluation_id') == budget)
    calls = [c for c in run['calls'] if c['candidate_id'] <= end['candidate_id']]
    ev = [e for e in run['events'] if e['candidate_id'] <= end['candidate_id']]
    start = datetime.fromisoformat(run['calls'][0]['ts'])
    return {'llm_calls': len(calls), 'invalid': sum(e['status'] == 'invalid_output' for e in ev),
            'eval_failed': sum(e['status'] == 'eval_failed' for e in ev),
            'repair_evaluations': sum(bool(e.get('repair_of')) and e.get('evaluation_id') is not None for e in ev),
            'wall_seconds': (datetime.fromisoformat(end['ts']) - start).total_seconds(),
            'prompt_tokens': sum((c.get('usage') or {}).get('prompt_tokens', 0) for c in calls),
            'completion_tokens': sum((c.get('usage') or {}).get('completion_tokens', 0) for c in calls)}


def diagnostics(run, budget=None):
    ev = run['events']
    if budget:
        boundary = next(e['candidate_id'] for e in ev if e.get('evaluation_id') == budget)
        ev = [e for e in ev if e['candidate_id'] <= boundary]
    nodes = run['nodes']; bycall = {c['call_id']: c for c in run['calls']}
    counts = Counter(); groups = defaultdict(Counter); invalid = Counter(); examples = {}; requests = Counter()
    last_improve = None; parent_selections = Counter(); equal_scores = Counter(); prompts = Counter()
    for event in ev:
        op = event['operator']; counts[event['status']] += 1
        repair = bool(event.get('repair_of')); counts['repairs'] += repair
        if event.get('frontier_improved') or last_improve is None and event['status'] == 'ok':
            last_improve = event.get('evaluation_id')
        selected = event.get('parent_selected', False)
        if selected:
            parent_selections[event['parent_id']] += 1
            counts['normal_development'] += 1
            groups['normal_' + op]['attempts'] += 1
            refs = event.get('reference_ids', [])
            counts['reference_cards_shown'] += len(refs)
            if op in ('Pivot','Fuse'):
                counts['normal_cross_branch'] += 1
                counts['cross_branch_without_read'] += not bool(event.get('context_reads') and event.get('loaded_reference_id') is not None)
                counts['cross_branch_declared_donor'] += event.get('donor_id') is not None
        if event.get('context_reads'):
            first = bycall.get(f"{event['candidate_id']}:1", {})
            payload = response_object(first.get('response', '')) or {}
            kind = ('reference' if event.get('loaded_reference_id') is not None else
                    'trials_nonempty' if event.get('trial_ids') else 'trials_empty')
            requests[kind] += 1
        call = bycall.get(f"{event['candidate_id']}:{event.get('llm_calls', 1)}")
        if call is None:
            matches = [c for c in run['calls'] if c['candidate_id'] == event['candidate_id']]
            call = matches[-1] if matches else {}
        if selected and op in ('Refine','Tune'):
            prompts[event['parent_id'],op,call.get('prompt')] += 1
        if event['status'] == 'invalid_output':
            text = THINK_BLOCK_RE.sub('', call.get('response', '')).strip()
            payload = response_object(text)
            reason = event.get('reason', '')
            if payload is not None:
                category = 'edit_contract' if reason.startswith('edit_error') else 'valid_json_invalid_code_or_fields'
            elif text.startswith('{') or text.startswith('```json'):
                category = 'json_envelope_not_decoded'
            elif '```json' in text:
                category = 'prose_or_mixed_blocks_before_json'
            else:
                category = 'other_non_json_or_code_failure'
            invalid[category] += 1
            if category not in examples:
                examples[category] = {'run':run['lane']['run_name'],'candidate_id':event['candidate_id'],
                                     'reason':reason,'prefix':text[:220],'suffix':text[-120:]}
        if event['status'] != 'ok':
            continue
        node = nodes[event['node_id']]
        equal_scores[node['fitness']] += 1
        parent = nodes.get(event['parent_id'])
        if parent is None:
            continue
        own = groups[op]; own['valid'] += 1
        improved = node['fitness'] > parent['fitness'] + 1e-10
        frontier = bool(event.get('frontier_improved'))
        tie = abs(node['fitness'] - parent['fitness']) < 1e-10
        copied = code_key(node['code']) == code_key(parent['code'])
        for key in (op, 'repair' if repair else 'normal', 'mode_'+str(event.get('output_mode')),
                    'context' if event.get('context_reads') else 'no_context'):
            group = groups[key]
            if key != op:group['valid'] += 1
            group['above_parent'] += improved; group['above_frontier'] += frontier
            group['equal_parent_fitness'] += tie; group['exact_parent_copy'] += copied
        if selected:
            groups['normal_'+op]['valid'] += 1
            groups['normal_'+op]['above_parent'] += improved
            groups['normal_'+op]['above_frontier'] += frontier
        if event.get('donor_id') is not None:
            donor = nodes[event['donor_id']]
            own['with_declared_donor'] += 1
            own['above_both'] += node['fitness'] > max(parent['fitness'], donor['fitness']) + 1e-10
            own['exact_donor_copy'] += code_key(node['code']) == code_key(donor['code'])
        refs = [nodes[i]['fitness'] for i in event.get('reference_ids', []) if i in nodes]
        if op in ('Pivot','Fuse') and refs:
            own['with_refs'] += 1
            own['above_host_and_best_reference'] += node['fitness'] > max(parent['fitness'],max(refs)) + 1e-10
    counts['repeated_local_prompt_after_first'] = sum(n-1 for n in prompts.values())
    counts['local_prompt_count'] = sum(prompts.values())
    return {'counts':dict(counts),'groups':{k:dict(v) for k,v in groups.items()},'invalid_categories':dict(invalid),
            'invalid_examples':examples,'read_requests':dict(requests),'last_frontier_evaluation':last_improve,
            'score_mode_count':equal_scores.most_common(1),'parent_top5':parent_selections.most_common(5)}


def main():
    versions={'r1':load_batch('20260923_v1013'),'r2':load_batch('20260924_v1013r2', LIMITS)}
    tasks=sorted({t for t,_ in versions['r2']});comparison=[];all_diag={}; prefixes=[]
    for task in tasks:
        common=min(max(e.get('evaluation_id') or 0 for e in versions[v][task,rep]['events'])
                   for v in versions for rep in range(1,5))//10*10
        for budget in sorted({50,common}):
            scores={v:[best_at(versions[v][task,rep],budget) for rep in range(1,5)] for v in versions}
            costs={v:[prefix_cost(versions[v][task,rep],budget) for rep in range(1,5)] for v in versions}
            comparison.append({'task':task,'budget':budget,'scores':scores,'mean':{v:statistics.mean(x) for v,x in scores.items()},
                               'median':{v:statistics.median(x) for v,x in scores.items()},
                               'r2_improvement_pct':100*(statistics.mean(scores['r2'])-statistics.mean(scores['r1']))/abs(statistics.mean(scores['r1'])),
                               'cost_totals':{v:{k:sum(c[k] for c in cs) for k in cs[0]} for v,cs in costs.items()},
                               'cost_per_lane':costs})
    for v, runs in versions.items():
        all_diag[v]={}
        for (task,rep), run in runs.items():
            all_diag[v][run['lane']['run_name']]={'task':task,'rep':rep,'all':diagnostics(run),
                                                 'E50':diagnostics(run,50)}
            prefixes.append({'version':v,'run':run['lane']['run_name'],
                             'last_candidate':run['events'][-1]['candidate_id']})
    output={'status':'partial','boundary_source':'progress_0121_protocol.json','comparison':comparison,'diagnostics':all_diag,'prefixes':prefixes}
    path=HERE/'progress_0121_diagnostics.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(comparison,ensure_ascii=False,indent=2))
    for v in versions:
        for scope in ('E50','all'):
            groups=defaultdict(Counter);counts=Counter();invalid=Counter();reads=Counter()
            for run in all_diag[v].values():
                d=run[scope];counts.update(d['counts']);invalid.update(d['invalid_categories']);reads.update(d['read_requests'])
                for k,vals in d['groups'].items():groups[k].update(vals)
            print(json.dumps({'version':v,'scope':scope,'counts':dict(counts),'invalid':dict(invalid),'reads':dict(reads),
                              'groups':{k:dict(c) for k,c in groups.items()}},ensure_ascii=False))


if __name__=='__main__':main()
