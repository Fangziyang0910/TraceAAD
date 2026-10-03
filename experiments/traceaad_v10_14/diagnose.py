"""Freeze append-only journal boundaries and measure search behavior, without LLM calls."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


def snapshot(root, output):
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(root.glob('*/*/search.jsonl'))
    boundaries = [(p, p.stat().st_size) for p in paths]
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(), 'runs': []}
    for path, limit in boundaries:
        config = json.loads((path.parent / 'run_config.json').read_text())
        name = f"{config['task']}_r{config['repeat']}"
        tables = {k: {} for k in ('anchor', 'attempt', 'request', 'revision', 'session', 'artifact')}
        calls, digest, used, state_line = {}, hashlib.sha256(), 0, None
        with path.open('rb') as f:
            while used < limit:
                line = f.readline(limit - used)
                if not line.endswith(b'\n'):
                    break
                digest.update(line)
                used += len(line)
                kind_match = re.match(rb'\{"kind":\s*"([^"]+)"', line)
                if not kind_match:
                    continue
                kind = kind_match[1].decode()
                if kind == 'state':
                    state_line = line
                elif kind in tables:
                    row = json.loads(line)['data']
                    tables[kind][row['id']] = row
                elif kind == 'call':
                    row = json.loads(line)
                    calls[row['request_id']] = row
        facts = {'config': config, 'tables': tables, 'calls': calls,
                 'state': json.loads(state_line)['state'] if state_line else None}
        (output / f'{name}.json').write_text(json.dumps(facts, ensure_ascii=False))
        manifest['runs'].append({'name': name, 'path': str(path), 'bytes': used,
                                 'sha256': digest.hexdigest()})
        print(name, len(tables['attempt']), flush=True)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))


def analyze(path):
    data = json.loads(path.read_text())
    t = data['tables']
    anchors = t['anchor']
    attempts = list(t['attempt'].values())
    requests = t['request']
    unique = {}
    for a in anchors.values():
        unique.setdefault(a['artifact_id'], a)
    used = Counter(anchors[str(a['parent_id'])]['artifact_id'] for a in attempts if a.get('parent_id'))
    n = len(attempts)
    mature = [a for a in unique.values() if a['id'] <= n-100 and a.get('parent_id')]
    worse = [a for a in mature if a['fitness'] < anchors[str(a['parent_id'])]['fitness']-1e-6]
    result = {'task': data['config']['task'], 'repeat': data['config']['repeat'],
              'backend': data['config']['backend'], 'completed': n,
              'status': dict(Counter(a['status'] for a in attempts)), 'unique': len(unique),
              'mature': len(mature), 'mature_used': sum(used[a['artifact_id']] > 0 for a in mature),
              'worse_mature': len(worse), 'worse_used': sum(used[a['artifact_id']] > 0 for a in worse),
              'actions': {}, 'idea': Counter(), 'context': Counter(), 'curve': {},
              'empty_examples': [], 'parent_top10_fraction': sum(v for _, v in used.most_common(10))/max(1, sum(used.values()))}
    best, last = None, 0
    for a in attempts:
        if a.get('fitness') is not None and (best is None or a['fitness'] > best+1e-6):
            best, last = a['fitness'], a['id']
        if a['id'] in (100, 200, 300, 500, 700, 1000):
            result['curve'][a['id']] = best
        action = a.get('executed_action') or a.get('scope')
        ac = result['actions'].setdefault(action, Counter())
        ac['n'] += 1
        ac[a['status']] += 1
        ac['global_improvements'] += int(a.get('global_gain', 0) > 1e-6)
        if a.get('parent_id') and a.get('fitness') is not None:
            ac['parent_improvements'] += int(a['fitness'] > anchors[str(a['parent_id'])]['fitness']+1e-6)
        if a.get('contract_diagnostic'):
            ac['contract_diagnostic'] += 1
        if not a.get('request_id'):
            continue
        call = data['calls'].get(str(a['request_id']), {})
        req = requests.get(str(a['request_id']), {})
        result['context']['requests'] += 1
        included, dropped = req.get('evidence_ids', []), req.get('dropped_evidence_ids', [])
        result['context']['no_evidence'] += int(not included)
        result['context']['formation_included'] += int(any(x.startswith('revision:') for x in included))
        result['context']['direct_included'] += int(any(x.startswith('attempt:') for x in included))
        result['context']['formation_dropped'] += sum(x.startswith('revision:') for x in dropped)
        result['context']['transfer_fallback'] += int(bool(req.get('reference_fallback')))
        if a['status'] not in ('ok', 'duplicate'):
            continue
        counts = result['idea']
        counts['valid'] += 1
        if a.get('idea', '').strip():
            counts['present'] += 1
        else:
            counts['empty'] += 1
            raw = call.get('response', '')
            category = ('idea_word_present' if re.search(r'\bidea\b', raw, re.I)
                        else 'prose_before_code' if raw.split('```')[0].strip().lower() not in ('', 'final implementation:', 'final code:')
                        else 'no_prose_before_code')
            counts[category] += 1
            if len(result['empty_examples']) < 4:
                result['empty_examples'].append({'candidate': a['id'], 'category': category,
                                                  'head': raw[:600], 'tail': raw[-600:]})
    result.update(best=best, last_improvement=last, plateau=n-last)
    result['parent_count_max'] = max(used.values(), default=0)
    result['range'] = [min(a['fitness'] for a in unique.values()), max(a['fitness'] for a in unique.values())] if unique else []
    if unique:
        maximum = max(a['fitness'] for a in unique.values())
        result['more_than_one_below_best'] = sum(a['fitness'] < maximum-1 for a in unique.values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--snapshot', type=Path, required=True)
    args = parser.parse_args()
    if args.root:
        if (args.snapshot / 'manifest.json').exists():
            raise ValueError('snapshot already exists')
        snapshot(args.root, args.snapshot)
    report = [analyze(args.snapshot / f"{r['name']}.json")
              for r in json.loads((args.snapshot / 'manifest.json').read_text())['runs']]
    (args.snapshot / 'analysis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    for row in report:
        print({k: row[k] for k in ('task', 'repeat', 'completed', 'best', 'plateau', 'idea', 'context')})


if __name__ == '__main__':
    main()
