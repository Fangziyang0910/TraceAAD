"""Descriptive paired summaries; attempts are not independent historical lineages."""

import argparse
from collections import Counter
import csv
from pathlib import Path
import statistics

from traceaad.storage import read_json, write_json
from .run import ARMS, digest


def outcome(child, parent):
    if parent.get('fitness') is None:
        raise ValueError('invalid parent: paired comparison is undefined')
    score = child.get('fitness')
    gain = (parent['fitness'] - score) / abs(parent['fitness']) if score is not None else None
    return dict(valid=score is not None, improved=gain is not None and gain > 1e-9,
                relative_gain=gain, fitness=score)


def summarize(rows):
    result = dict(n=len(rows), generated=sum(r['status'] == 'generated' for r in rows),
                  not_applicable=sum(r['status'] == 'not_applicable' for r in rows),
                  delivery_failed=sum(r['status'] == 'delivery_failed' for r in rows),
                  faithful=sum(r.get('faithful') is True for r in rows),
                  audited=sum(r.get('faithful') is not None for r in rows))
    for split in ('train', 'heldout'):
        gains = [r[f'{split}_relative_gain'] for r in rows if r[f'{split}_relative_gain'] is not None]
        result[split] = dict(valid=len(gains), improved=sum(r[f'{split}_improved'] for r in rows),
                             mean_gain=statistics.fmean(gains) if gains else None,
                             median_gain=statistics.median(gains) if gains else None,
                             min_gain=min(gains) if gains else None,
                             max_gain=max(gains) if gains else None)
    return result


def analyze(out):
    data = read_json(out / 'inputs.json')
    audit = {}
    audit_errors = []
    for path in sorted((out / 'audits').glob('*.json')):
        row = read_json(path)
        if row.get('error'):
            audit_errors.append(dict(file=path.name, error=row['error']))
            continue
        for label, cell_id in row['mapping'].items():
            verdict = row['result']['candidates'][label]
            audit[cell_id] = dict(
                audit_applicable=row['result']['applicable_to_parent'],
                implements=verdict['implements_operation'],
                faithful=verdict['implements_operation'] and verdict['preserves_required']
                         and not verdict['unrelated_changes'],
                audit_evidence=verdict['evidence'])
    rows, missing = [], []
    for cell in data['schedule']:
        candidate = read_json(out / 'candidates' / f'{cell["id"]}.json')
        if candidate is None:
            missing.append(cell['id'])
            continue
        row = {**cell, 'status': candidate['status'], **audit.get(cell['id'], {})}
        for split in ('train', 'heldout'):
            parent = read_json(out / 'evaluations' / f'parent_{cell["parent"]}_{split}.json')
            result = read_json(out / 'evaluations' / f'{cell["id"]}_{split}.json')
            if parent is None or (candidate.get('code') and result is None):
                missing.append(f'{cell["id"]}:{split}')
                continue
            if result is not None and result['code_sha256'] != digest(candidate['code']):
                raise ValueError(f'evaluation/source mismatch: {cell["id"]}')
            values = outcome(result or {}, parent)
            row.update({f'{split}_{k}': v for k, v in values.items()})
        if 'heldout_valid' in row and 'train_valid' in row:
            rows.append(row)
    costs = {}
    for path in sorted((out / 'calls').glob('*.json')):
        category = ('induction' if path.stem.startswith('induce_') else
                    'audit' if path.stem.startswith('audit_') else path.stem.rsplit('_', 1)[-1])
        c = costs.setdefault(category, Counter())
        for call in read_json(path)['details']['calls']:
            c['requests'] += 1
            c['errors'] += bool(call.get('error'))
            usage = call.get('usage') or {}
            for field in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                if usage.get(field) is not None:
                    c[field] += usage[field]
                else:
                    c[f'missing_{field}'] += 1
            c[f'endpoint:{(call.get("scheduler") or {}).get("endpoint", "unknown")}'] += 1
    aggregate = {arm: summarize([r for r in rows if r['arm'] == arm]) for arm in ARMS}
    by_parent = {p['id']: {arm: summarize([r for r in rows if r['parent'] == p['id'] and r['arm'] == arm])
                          for arm in ARMS} for p in data['parents']}
    by_source = {s['id']: {arm: summarize([r for r in rows if r['source'] == s['id'] and r['arm'] == arm])
                          for arm in ARMS} for s in data['sources']}
    paired = {}
    for a, b in [('advice', 'raw'), ('constrained', 'advice'), ('constrained', 'raw')]:
        pairs = []
        lookup = {(r['source'], r['parent'], r['repeat'], r['arm']): r for r in rows}
        for r in rows:
            if r['arm'] != a:
                continue
            other = lookup.get((r['source'], r['parent'], r['repeat'], b))
            if other:
                pairs.append(dict(parent=r['parent'],
                                  improvement_difference=int(r['heldout_improved']) - int(other['heldout_improved'])))
        paired[f'{a}_minus_{b}'] = dict(
            n_pairs=len(pairs),
            improvement_rate_difference=statistics.fmean(p['improvement_difference'] for p in pairs) if pairs else None,
            by_parent={p['id']: statistics.fmean(x['improvement_difference'] for x in pairs if x['parent'] == p['id'])
                       for p in data['parents'] if any(x['parent'] == p['id'] for x in pairs)})
    summary = dict(status='complete' if len(rows) == len(data['schedule']) and not missing else 'incomplete',
                   expected=len(data['schedule']), evaluated=len(rows), missing=missing,
                   audit_count=len(audit), audit_errors=audit_errors, arms=aggregate,
                   by_parent=by_parent, by_source=by_source, paired=paired, costs=costs,
                   interpretation='Descriptive pilot: 6 recipients from 2 runs, 3 researcher-selected transforms. '
                   'Valid-only gains include no-ops but omit invalid candidates; improvement counts include all scheduled attempts. '
                   'Actual total tokens differ. No significance or equivalence claim.')
    write_json(out / 'analysis.json', summary)
    if rows:
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with (out / 'results.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    print(summary['status'], len(rows), '/', len(data['schedule']))
    for arm, s in aggregate.items():
        print(arm, s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    analyze(parser.parse_args().out)
