"""Copy the completed V10.6 suffix after E1 without modifying live runs."""
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from experiments.traceaad_refine_e1.prepare import ROOT, dump, records

DEFAULT = ROOT / 'experiments/traceaad_e2_a/raw/traceaad_e2_a_20260907'
SOURCE = ROOT / 'experiments/traceaad_v10_6/results/batch_20260906_215231_revised.json'
E1 = ROOT / 'experiments/traceaad_refine_e1/raw/refine_e1_20260907/snapshot.json'


def prepare_snapshot(out=DEFAULT):
    if (out / 'snapshot.json').exists():
        return json.loads((out / 'snapshot.json').read_text())
    batch = json.loads(SOURCE.read_text())
    e1 = json.loads(E1.read_text())
    prior = {r['run_name']: r for r in e1['runs']}
    runs = []
    for spec in batch['plan']:
        source = SOURCE.parent / spec['task'] / spec['run_name']
        state = json.loads((source / 'tree_state.json').read_text())
        cutoff = state['completed_attempts']
        events = [e for e in records(source / 'events.jsonl') if e['candidate_id'] <= cutoff]
        receipts = [e for e in records(source / 'evaluations.jsonl') if e['candidate_id'] <= cutoff]
        assert [e['candidate_id'] for e in events] == list(range(1, cutoff + 1))
        old = prior[spec['run_name']]['cutoff_attempt']
        assert cutoff > old
        suffix = [e for e in events if e['candidate_id'] > old]
        nodes = {n['id']: n for n in state['nodes']}
        born = {e['node_id']: e['candidate_id'] for e in events if e.get('node_id') is not None}
        assert set(born) == set(nodes)
        for event in events:
            pid = event.get('parent_id')
            if pid is not None:
                assert born[pid] < event['candidate_id']
                assert nodes[pid]['fitness'] == event['parent_fitness']
            conditional = event.get('selection', {}).get('operator_conditional')
            if event['requested_operator'] in ['Refine', 'Pivot', 'Fuse']:
                assert conditional and abs(sum(conditional.values()) - 1) < 1e-10
        target = out / 'snapshot' / spec['run_name']
        dump(target / 'tree_state.json', state)
        dump(target / 'run_config.json', json.loads((source / 'run_config.json').read_text()))
        for name, rows in [('events.jsonl', events), ('evaluations.jsonl', receipts)]:
            (target / name).write_text(''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in rows))
        runs.append({
            **spec,
            'source': str(source),
            'e1_cutoff_attempt': old,
            'e1_cutoff_evaluation': prior[spec['run_name']]['cutoff_evaluation'],
            'cutoff_attempt': cutoff,
            'cutoff_evaluation': state['budget_used'],
            'suffix_attempts': len(suffix),
            'suffix_requested_executed': dict(Counter(
                f"{e['requested_operator']}->{e['operator']}" for e in suffix)),
            'nodes': len(nodes),
        })
    result = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'source_batch': str(SOURCE),
        'e1_snapshot': str(E1),
        'runs': runs,
    }
    dump(out / 'snapshot.json', result)
    dump(out / 'e2a_config.json', json.loads((Path(__file__).with_name('e2a_config.json')).read_text()))
    return result


if __name__ == '__main__':
    snapshot = prepare_snapshot()
    print(json.dumps({
        'runs': len(snapshot['runs']),
        'suffix_attempts': sum(r['suffix_attempts'] for r in snapshot['runs']),
        'cutoff_evaluations': sum(r['cutoff_evaluation'] for r in snapshot['runs']),
    }))
