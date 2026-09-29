"""Additional descriptive checks on frozen V10.14-2 facts; never rescore code."""

import argparse
import ast
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import warnings


def audit(path):
    data = json.loads(path.read_text())
    tables = data['tables']
    anchors, artifacts = tables['anchor'], tables['artifact']
    ideas, actions, contexts = Counter(), defaultdict(Counter), Counter()
    examples, seen, keys = [], set(), set()
    for anchor in anchors.values():
        if anchor['artifact_id'] in seen:
            continue
        seen.add(anchor['artifact_id'])
        code = artifacts[anchor['artifact_id']]['code']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SyntaxWarning)
            keys.add(ast.dump(ast.parse(code), include_attributes=False))
    for attempt in tables['attempt'].values():
        parent = anchors.get(str(attempt.get('parent_id')))
        child = anchors.get(str(attempt.get('anchor_id')))
        request = tables['request'].get(str(attempt.get('request_id')), {})
        if parent and request:
            contexts['parent_requests'] += 1
            revision = parent.get('revision_id')
            if revision:
                contexts['parent_has_formation'] += 1
                contexts['immediate_formation_seen'] += f'revision:{revision}' in request.get('evidence_ids', [])
        if attempt['channel'] == 'main':
            row = actions[attempt.get('executed_action', attempt['scope'])]
            row['n'] += 1
            row['contract_deviation'] += bool(attempt.get('contract_deviation'))
            if parent:
                row['parent_over_one_below_best'] += parent['fitness'] < attempt['global_best_before']-1
            if child and parent:
                row['valid'] += 1
                row['same_score'] += abs(child['fitness']-parent['fitness']) <= 1e-6
                row['same_probes'] += bool(child.get('profile')) and child['profile'] == parent.get('profile')
                row['same_score_and_probes'] += abs(child['fitness']-parent['fitness']) <= 1e-6 and child.get('profile') == parent.get('profile')
        if attempt['status'] not in ('ok', 'duplicate'):
            continue
        ideas['valid'] += 1
        if attempt.get('idea', '').strip():
            ideas['present'] += 1
            continue
        ideas['empty'] += 1
        call = data['calls'][str(attempt['request_id'])]
        raw = call['response']
        outside = re.sub(r'```[^\n]*\n.*?```', '\n', raw, flags=re.S)
        if re.search(r'(?im)^\s*(?:#+\s*)?(?:\*\*)?Idea(?:\*\*)?\s*(?::|\n)', outside):
            category = 'explicit_idea_outside_code'
        else:
            prose = re.sub(r'(?im)^\s*(?:#+\s*)?(?:\*\*)?Final (?:implementation|code|candidate)(?:\*\*)?:?(?:\*\*)?\s*$', '', outside)
            category = 'unlabelled_outside_text' if prose.strip() else 'no_outside_rationale'
        ideas[category] += 1
        if len([e for e in examples if e['category'] == category]) < 2:
            examples.append({'id': attempt['id'], 'category': category, 'outside_text': outside[:1800]})
    state = data.get('state') or {}
    return {'run': path.stem, 'task': data['config']['task'], 'backend': data['config']['backend'],
            'raw_unique_sources': len(seen), 'syntax_unique_sources': len(keys),
            'ideas': ideas, 'actions': actions, 'context': contexts, 'examples': examples,
            'tokens': state.get('ledger', {}).get('tokens'), 'elapsed': state.get('elapsed')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    args = parser.parse_args()
    rows = [audit(args.snapshot / f"{r['name']}.json")
            for r in json.loads((args.snapshot / 'manifest.json').read_text())['runs']]
    (args.snapshot / 'audit.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    for task in sorted({r['task'] for r in rows}):
        group = [r for r in rows if r['task'] == task]
        print(task, 'idea', sum((Counter(r['ideas']) for r in group), Counter()),
              'context', sum((Counter(r['context']) for r in group), Counter()))


if __name__ == '__main__':
    main()
