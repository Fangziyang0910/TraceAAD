"""Copy a read-only V10.6 prefix without modifying live journals."""
import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'experiments/traceaad_refine_e1/raw/refine_e1_20260907'


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    tmp.replace(path)


def records(path):
    data = path.read_bytes()
    end = data.rfind(b'\n') + 1
    return [json.loads(line) for line in data[:end].splitlines() if line.strip()]


def prepare_snapshot(out):
    if (out / 'snapshot.json').exists():
        return json.loads((out / 'snapshot.json').read_text())
    manifest = ROOT / 'experiments/traceaad_v10_6/results/batch_20260906_215231_revised.json'
    batch = json.loads(manifest.read_text())
    runs = []
    for spec in batch['plan']:
        source = manifest.parent / spec['task'] / spec['run_name']
        state = json.loads((source / 'tree_state.json').read_text())
        assert state['version'] == 106
        assert state['mechanism']['generation'] == 'idea_code_then_implementation_idea'
        cutoff = state['completed_attempts']
        events = [e for e in records(source / 'events.jsonl') if e['candidate_id'] <= cutoff]
        assert [e['candidate_id'] for e in events] == list(range(1, cutoff + 1))
        receipts = [e for e in records(source / 'evaluations.jsonl') if e['candidate_id'] <= cutoff]
        nodes = {n['id']: n for n in state['nodes']}
        born = {e['node_id']: e['candidate_id'] for e in events if e.get('node_id') is not None}
        assert set(born) == set(nodes)
        refinements = [e for e in events if e['requested_operator'] == e['operator'] == 'Refine']
        for e in events:
            pid = e.get('parent_id')
            if pid is not None:
                assert born[pid] < e['candidate_id']
                assert nodes[pid]['fitness'] == e['parent_fitness']
            if e.get('parent_delta') is not None:
                assert abs(e['parent_delta'] - (e['fitness'] - e['parent_fitness'])) < 1e-10
        target = out / 'snapshot' / spec['run_name']
        # Keep the inputs necessary for replay; source journals may keep growing.
        dump(target / 'tree_state.json', state)
        dump(target / 'run_config.json', json.loads((source / 'run_config.json').read_text()))
        for name, rows in [('events.jsonl', events), ('evaluations.jsonl', receipts)]:
            (target / name).write_text(''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in rows))
        counts = Counter(e['parent_id'] for e in refinements)
        runs.append(dict(spec, source=str(source), cutoff_attempt=cutoff,
            cutoff_evaluation=state['budget_used'], nodes=len(nodes), refine=len(refinements),
            refine_parents=len(counts), parents_one_or_two=sum(v <= 2 for v in counts.values()),
            improved=sum(bool(e.get('parent_improved')) for e in refinements),
            status_counts=dict(Counter(e['status'] for e in refinements)),
            fallback_refine=sum(e['requested_operator']=='Fuse' and e['operator']=='Refine' for e in events)))
    result = dict(created_at=datetime.now(timezone.utc).isoformat(), source_batch=str(manifest), runs=runs)
    dump(out / 'snapshot.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=DEFAULT)
    args = parser.parse_args()
    result = prepare_snapshot(args.out)
    print(json.dumps({'runs':len(result['runs']), 'refine':sum(r['refine'] for r in result['runs']),
                      'parents':sum(r['refine_parents'] for r in result['runs']),
                      'improved':sum(r['improved'] for r in result['runs'])}))
