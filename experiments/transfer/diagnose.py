"""Post-hoc diagnostics; never changes the frozen primary outcomes."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from traceaad.storage import read_json, write_json
from .run import ARMS, canonical


def diagnose(out):
    data = read_json(out / 'inputs.json')
    sources = {s['id']: s for s in data['sources']}
    endpoint = defaultdict(Counter)
    counts = {a: Counter() for a in ARMS}
    copies, disagreement = [], []
    constrained = defaultdict(list)
    for cell in data['schedule']:
        candidate = read_json(out / 'candidates' / f'{cell["id"]}.json')
        assert candidate is not None
        result = read_json(out / 'evaluations' / f'{cell["id"]}_heldout.json', {})
        parent = read_json(out / 'evaluations' / f'parent_{cell["parent"]}_heldout.json')['fitness']
        donor = read_json(out / 'evaluations' / f'source_{cell["source"]}_after_heldout.json')['fitness']
        score = result.get('fitness')
        c = counts[cell['arm']]
        c['n'] += 1
        c['direct_copy_improves'] += donor < parent - 1e-9
        c['beats_both'] += score is not None and score < min(parent, donor) - 1e-9
        c['changed_valid'] += candidate['status'] == 'generated' and score is not None
        c['changed_improves'] += candidate['status'] == 'generated' and score is not None and score < parent - 1e-9
        if candidate.get('code') and canonical(candidate['code']) == canonical(sources[cell['source']]['child']['code']):
            copies.append(cell['id'])
        calls = read_json(out / 'calls' / f'{cell["id"]}.json')['details']['calls']
        route = (calls[-1].get('scheduler') or {}).get('endpoint', 'unknown')
        e = endpoint[f'{cell["arm"]}:{route}']
        e['n'] += 1
        e['delivery_failed'] += candidate['status'] == 'delivery_failed'
        e['improved'] += score is not None and score < parent - 1e-9
        if cell['arm'] == 'constrained' and candidate.get('reply') is not None:
            constrained[(cell['source'], cell['parent'])].append(
                (cell['id'], candidate['reply']['applicable']))
    for pair, values in constrained.items():
        if len(values) == 2 and values[0][1] != values[1][1]:
            disagreement.append(dict(source=pair[0], parent=pair[1], replies=values))
    control_cost = Counter()
    for path in (out / 'diagnostic_no_history/calls').glob('*.json'):
        for call in read_json(path)['details']['calls']:
            control_cost['requests'] += 1
            for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                if (call.get('usage') or {}).get(key) is not None:
                    control_cost[key] += call['usage'][key]
                else:
                    control_cost[f'missing_{key}'] += 1
    assert all(c['n'] == 36 for c in counts.values())
    assert len({c['direct_copy_improves'] for c in counts.values()}) == 1
    result = dict(status='posthoc descriptive diagnostics', arms=counts, endpoint=endpoint,
                  exact_donor_copies=copies, applicability_disagreement=disagreement,
                  no_history_cost=control_cost)
    write_json(out / 'diagnostics.json', result)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    diagnose(parser.parse_args().out)
