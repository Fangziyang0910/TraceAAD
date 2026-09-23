"""Read-only, checkpoint-bounded protocol audit of an r2 or r3 batch.

Writes only an explicitly requested report outside the experiment journals.
No LLM requests, evaluator calls, resume, or mutations of live state occur.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_prefix(path):
    if not path.exists():
        return []
    data = path.read_bytes()
    data = data[:data.rfind(b'\n') + 1]
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def record_hash(records):
    return sha(json.dumps(records, sort_keys=True, ensure_ascii=False).encode())


def audit(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    runtime = Path(manifest['runtime'])
    frozen = json.loads((runtime / 'runtime_manifest.json').read_text())
    drift = [name for name, expected in frozen['files'].items()
             if not (runtime / name).is_file() or sha((runtime / name).read_bytes()) != expected]
    if drift:
        raise ValueError(f'frozen runtime changed: {drift}')
    sys.path.insert(0, str(runtime))
    from traceaad.v10_13.parsing import code_hash, parse_candidate, response_object, template_target
    from traceaad.v10_13.selection import code_key

    pane_text = subprocess.check_output(
        ['tmux', 'list-panes', '-a', '-F', '#{session_name}\t#{pane_pid}'], text=True)
    panes = {session: int(pid) for session, pid in
             (line.split('\t') for line in pane_text.splitlines())}
    totals = Counter()
    statuses, operators, modes, reasons, by_operator = Counter(), Counter(), Counter(), Counter(), {}
    examples, runs, violations = {}, [], []
    for lane in manifest['plan']:
        directory = manifest_path.parent / lane['task'] / lane['run_name']
        state_data = (directory / 'tree_state.json').read_bytes()
        state = json.loads(state_data)
        boundary = state['completed_candidates']
        pending = state.get('pending') or {}
        events = [e for e in read_prefix(directory / 'events.jsonl') if e['candidate_id'] <= boundary]
        # A committing checkpoint can precede the final journal projection.
        effective = boundary - int(pending.get('stage') == 'committing' and
                                   not any(e['candidate_id'] == boundary for e in events))
        events = [e for e in events if e['candidate_id'] <= effective]
        node_ids = {e['node_id'] for e in events if e.get('node_id') is not None}
        nodes = {n['id']: n for n in state['nodes'] if n['id'] in node_ids}
        calls = [c for c in read_prefix(directory / 'llm_calls.jsonl') if c['candidate_id'] <= effective]
        receipts = [r for r in read_prefix(directory / 'evaluations.jsonl') if r['candidate_id'] <= effective]
        local = []

        def check(condition, message):
            if not condition:
                local.append(message)

        revision = state['mechanism']['revision']
        check(revision in ('v10.13-r2', 'v10.13-r3'), 'revision mismatch')
        check([e['candidate_id'] for e in events] == list(range(1, effective + 1)), 'candidate ID gap/duplicate')
        eval_ids = [e['evaluation_id'] for e in events if e.get('evaluation_id') is not None]
        check(eval_ids == list(range(1, len(eval_ids) + 1)), 'evaluation ID gap/duplicate')
        check([r['evaluation_id'] for r in receipts] == eval_ids, 'receipt/event evaluation mismatch')
        check(len({c['call_id'] for c in calls}) == len(calls), 'duplicate LLM call ID')
        check(len(node_ids) == len(nodes), 'event refers to missing checkpoint node')
        check(0 <= state['budget_used'] - len(eval_ids) <= 1, 'budget not explained by pending work')
        recorded_nodes = {n['id']: n for n in read_prefix(directory / 'nodes.jsonl') if n['id'] in node_ids}
        check(set(recorded_nodes) == node_ids, 'missing durable node record')
        pid = panes.get(lane['session'])
        live = False
        if pid:
            proc = Path('/proc') / str(pid)
            try:
                args = proc.joinpath('cmdline').read_bytes().split(b'\x00')
                live = lane['run_name'].encode() in args and proc.joinpath('comm').read_text().startswith('python')
                check(proc.joinpath('cwd').resolve() == runtime, 'process cwd differs from frozen runtime')
            except OSError:
                pass
        config = json.loads((directory / 'run_config.json').read_text())
        check(config['seed'] == lane['seed'] and config['backend'] == lane['backend'], 'seed/backend mismatch')
        check(config['method_params']['budget'] == 1000, 'formal budget mismatch')
        by_call = {c['call_id']: c for c in calls}
        by_receipt = {r['evaluation_id']: r for r in receipts}
        by_event = {e['candidate_id']: e for e in events}
        template = importlib.import_module(f"benchmarks.{lane['task']}.template").template_program
        interface, _ = template_target(template)
        for event in events:
            cid = event['candidate_id']
            tag = f'candidate {cid}: '
            statuses[event['status']] += 1
            operators[event['operator']] += 1
            modes[event.get('output_mode') or 'invalid'] += 1
            counts = by_operator.setdefault(event['operator'], Counter())
            counts['events'] += 1
            counts[event['status']] += 1
            if event.get('reason'):
                reasons[event['reason'].split(':', 1)[0]] += 1
            totals['context_reads'] += event.get('context_reads', 0)
            totals['events_with_loaded_reference'] += event.get('loaded_reference_id') is not None
            totals['reference_read_rounds'] += bool(event.get('context_reads') and event.get('loaded_reference_id') is not None)
            totals['declared_donors'] += event.get('donor_id') is not None
            totals['events_with_local_trials'] += bool(event.get('trial_ids'))
            totals['nonempty_local_read_rounds'] += bool(event.get('context_reads') and event.get('trial_ids'))
            totals['repairs'] += event.get('repair_of') is not None
            check(event.get('context_reads', 0) <= 1, tag + 'multiple context rounds')
            if event.get('repair_of'):
                original = by_event[event['repair_of']]
                check(not event['parent_selected'], tag + 'repair reselected parent')
                check((event['parent_id'], event['operator']) ==
                      (original['parent_id'], original['operator']), tag + 'repair changed search choice')
                totals['successful_repairs'] += event['status'] == 'ok'
            if event['parent_selected']:
                check(event['selection']['uniform_mass'] == .125, tag + 'wrong exploration mass')
                selection = event['selection']
                check(selection['parent_probability'] >= .125 / selection['population_size'],
                      tag + 'parent below exploration floor')
            if revision == 'v10.13-r3':
                check(event.get('context_reads', 0) == 0, tag + 'r3 unexpectedly read context')
                check(event['llm_calls'] == 1, tag + 'r3 unexpected multi-call proposal')
            eid = event.get('evaluation_id')
            if eid is not None:
                receipt = by_receipt.get(eid, {})
                check(receipt.get('candidate_id') == cid, tag + 'receipt points to another candidate')
                check(receipt.get('fitness') == event.get('fitness'), tag + 'receipt/event fitness mismatch')
            call = by_call.get(f"{cid}:{event['llm_calls']}")
            check(call is not None, tag + 'final response missing')
            if call is None:
                continue
            check(sha(call['prompt'].encode()) == event['prompt_hash'], tag + 'prompt hash mismatch')
            check(event['prompt_tokens'] <= state['mechanism']['max_input_tokens'], tag + 'input too large')
            check('```diff' not in call['prompt'], tag + 'historical diff unexpectedly included')
            if revision == 'v10.13-r3' and event['parent_selected']:
                check('base_hash:' not in call['prompt'], tag + 'r3 asks model to copy hash')
                check('"mode":"context"' not in call['prompt'], tag + 'r3 offers context tool')
                if event['operator'] == 'Fuse':
                    identity = event.get('loaded_reference_id')
                    check(event['reference_ids'] == ([identity] if identity is not None else []),
                          tag + 'r3 reference exposure inconsistent')
                    if identity is not None:
                        check(nodes[identity]['code'] in call['prompt'], tag + 'inline reference missing')
                else:
                    check(not event['reference_ids'], tag + 'r3 non-Fuse reference')
            if event.get('context_reads'):
                first = by_call.get(f'{cid}:1')
                request = response_object(first['response']) if first else None
                check(request is not None and request.get('mode') == 'context', tag + 'missing context request')
                check(call.get('context_round') == 1, tag + 'wrong response context round')
                totals['empty_local_read_rounds'] += bool(request and request.get('trials') and not event.get('trial_ids'))
                reference_id = event.get('loaded_reference_id')
                if reference_id is not None:
                    check(reference_id in event['reference_ids'], tag + 'read unavailable reference')
                    check(nodes[reference_id]['code'] in call['prompt'], tag + 'reference implementation missing from prompt')
            if event['parent_selected'] and event['operator'] in ('Pivot', 'Fuse'):
                refs = [nodes[i] for i in event['reference_ids']]
                check(len(refs) <= state['mechanism']['reference_count'], tag + 'too many references')
                check(len({code_key(n['code']) for n in refs}) == len(refs), tag + 'duplicate reference implementation')
                check(all(code_key(n['code']) != code_key(nodes[event['parent_id']]['code']) for n in refs),
                      tag + 'reference duplicates parent')
                counts['cards_with_idea'] += sum(bool(n.get('idea') or n.get('idea_fields')) for n in refs)
                counts['cards'] += len(refs)
                if refs:
                    examples.setdefault('reference_shortlist', {'run': lane['run_name'], 'candidate_id': cid,
                                        'reference_ids': event['reference_ids'], 'sources':event['selection'].get('reference_sources')})
            for identity in event.get('trial_ids', []):
                check(nodes[identity]['parent_id'] == event['parent_id'], tag + 'trial is not exact-parent child')
            if event['status'] != 'ok':
                continue
            counts['valid_' + event['output_mode']] += 1
            node = nodes[event['node_id']]
            parent = nodes.get(event['parent_id'])
            parsed, error = parse_candidate(call['response'], call['finish_reason'], interface, template,
                                           base_code=parent['code'] if parent else None,
                                           allowed_donor_ids=event['reference_ids'])
            check(error is None and parsed.program_code == node['code'], tag + 'response does not reconstruct node')
            if parsed is not None:
                check(parsed.donor_id == event.get('donor_id'), tag + 'donor declaration mismatch')
            check(node.get('idea_fields', {}) == (parsed.idea_fields if parsed else None), tag + 'idea fields mismatch')
            totals['replayed_valid_candidates'] += 1
            totals['structured_idea_nodes'] += bool(node.get('idea_fields'))
            totals['missing_idea_nodes'] += not (node.get('idea') or node.get('idea_fields'))
            if parent:
                changed = code_key(parent['code']) != code_key(node['code'])
                totals['changed_parent_ast'] += changed
                counts['changed_parent_ast'] += changed
                counts['above_parent'] += node['fitness'] > parent['fitness']
                counts['above_frontier'] += bool(event.get('frontier_improved'))
                if event.get('donor_id') is not None:
                    donor = nodes[event['donor_id']]
                    counts['above_both'] += node['fitness'] > max(parent['fitness'],donor['fitness'])
                for kind, active in (('edit', event['output_mode'] == 'edit'),
                                     ('context_read', bool(event.get('context_reads'))),
                                     ('local_trial_read', bool(event.get('context_reads') and event.get('trial_ids'))),
                                     ('loaded_reference', event.get('loaded_reference_id') is not None),
                                     ('declared_donor', event.get('donor_id') is not None)):
                    if active:
                        examples.setdefault(kind, {'run': lane['run_name'], 'candidate_id': cid,
                            'operator': event['operator'], 'parent_id': parent['id'], 'node_id': node['id'],
                            'loaded_reference_id': event.get('loaded_reference_id'), 'donor_id': event.get('donor_id'),
                            'trial_ids': event.get('trial_ids'), 'parent_fitness': parent['fitness'],
                            'fitness':node['fitness'],'changed_parent_ast':changed,'code_sha256':code_hash(node['code'])})
        totals['settled_evaluations'] += len(eval_ids)
        all_reference_ids = {identity for event in events if event['parent_selected']
                             for identity in event['reference_ids']}
        totals['sum_distinct_references_per_lane'] += len(all_reference_ids)
        totals['llm_calls'] += len(calls)
        totals['prompt_tokens_reported'] += sum((c.get('usage') or {}).get('prompt_tokens', 0) for c in calls)
        totals['completion_tokens_reported'] += sum((c.get('usage') or {}).get('completion_tokens', 0) for c in calls)
        summary_path = directory / 'logs/run_summary.json'
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        check(not summary.get('status') or summary['status'] == 'finished', 'run stopped: '+str(summary.get('status')))
        runs.append({'run_name':lane['run_name'],'task':lane['task'],'repeat':lane['repeat'],
                     'backend':lane['backend'],'live':live,'summary_status':summary.get('status'),
                     'budget_used':state['budget_used'],'nodes':len(nodes),'effective_candidate_boundary':effective,
                     'pending_stage':pending.get('stage'),'events_prefix_sha256':record_hash(events),
                     'calls_prefix_sha256':record_hash(calls),'receipts_prefix_sha256':record_hash(receipts),
                     'checkpoint_sha256':sha(state_data),'violations':local})
        violations.extend(f"{lane['run_name']}: {message}" for message in local)
    return {'batch':manifest['batch'],'status':'partial','observed_at':datetime.now().astimezone().isoformat(timespec='seconds'),
            'scope':'checkpoint-bounded execution audit, not quality ranking or held-out evidence',
            'runtime':str(runtime),'frozen_git_base':frozen['git_base'],
            'source_identity':sha((runtime/'runtime_manifest.json').read_bytes()),'frozen_drift':drift,
            'totals':dict(totals),'statuses':dict(statuses),'operators':dict(operators),'modes':dict(modes),
            'failure_reasons':dict(reasons),'operator_diagnostics':{k:dict(v) for k,v in by_operator.items()},
            'examples':examples,'runs':runs,'violations':violations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = audit(args.manifest.resolve())
    if args.output:
        if args.output.exists():
            raise FileExistsError('Use a new snapshot filename; do not overwrite earlier evidence')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('batch','observed_at','totals','statuses','operators',
                                                 'modes','failure_reasons','violations')},ensure_ascii=False,indent=2))
    print(json.dumps({'live_runs':sum(r['live'] for r in report['runs']),
                      'budget_range':[min(r['budget_used'] for r in report['runs']),max(r['budget_used'] for r in report['runs'])],
                      'examples':report['examples']},ensure_ascii=False,indent=2))
    if report['violations']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
