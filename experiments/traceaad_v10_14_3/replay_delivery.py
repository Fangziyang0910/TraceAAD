"""Replay frozen responses without model calls, code execution or old-log edits."""

import argparse
from collections import Counter
from importlib import import_module
import json
from pathlib import Path
import warnings

from traceaad.v10_14_2.edits import parse_response as old_parse
from traceaad.v10_14_3.edits import parse_response as new_parse


def replay(path):
    data = json.loads(path.read_text())
    template = import_module(f"benchmarks.{data['config']['task']}.template").template_program
    counts, examples, mismatches = Counter(), [], []
    for attempt in data['tables']['attempt'].values():
        call = data['calls'].get(str(attempt.get('request_id')))
        if not call or call.get('error'):
            continue
        request = data['tables']['request'][str(attempt['request_id'])]
        if request['output_mode'] != 'full':
            raise ValueError('this frozen comparison expects full-module responses')
        outcomes = []
        for parser in (old_parse, new_parse):
            metadata = {}
            try:
                code, idea = parser(call['response'], call['finish_reason'], template, metadata=metadata)
                outcomes.append({'accepted': True, 'code': code, 'idea': idea, 'metadata': metadata})
            except (ValueError, SyntaxError, TypeError) as exc:
                outcomes.append({'accepted': False, 'code': getattr(exc, 'code', None), 'error': str(exc)})
        old, new = outcomes
        counts['responses'] += 1
        counts['old_accepted'] += old['accepted']
        counts['new_accepted'] += new['accepted']
        if old['accepted'] != new['accepted'] or old['code'] != new['code']:
            mismatches.append({'candidate': attempt['id'], 'old': old, 'new': new})
        if attempt['status'] not in ('ok', 'duplicate'):
            continue
        counts['valid'] += 1
        if not new['accepted']:
            counts['valid_rejected'] += 1
            continue
        was_empty, now_empty = not attempt.get('idea', '').strip(), not new['idea'].strip()
        counts['old_empty'] += was_empty
        counts['new_empty'] += now_empty
        if was_empty and not now_empty:
            kind = new['metadata']['idea_sources'][0]['kind']
            counts['recovered'] += 1
            counts[f'recovered_{kind}'] += 1
            if len(examples) < 8:
                examples.append({'candidate': attempt['id'], 'sources': new['metadata']['idea_sources'],
                                 'idea_excerpt': new['idea'][:800]})
        if not was_empty and now_empty:
            counts['old_present_now_missing'] += 1
    return {'run': path.stem, 'counts': counts, 'mismatches': mismatches, 'examples': examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('replay output already exists')
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SyntaxWarning)
        for run in json.loads((args.snapshot / 'manifest.json').read_text())['runs']:
            row = replay(args.snapshot / f"{run['name']}.json")
            rows.append(row)
            print(row['run'], dict(row['counts']), flush=True)
    total = dict(sum((r['counts'] for r in rows), Counter()))
    unchanged = all(not r['mismatches'] for r in rows)
    report = {'snapshot': str(args.snapshot), 'counts': total, 'code_and_acceptance_unchanged': unchanged,
              'runs': rows, 'scope': 'offline extraction only; no scoring or LLM calls'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({'counts': total, 'code_and_acceptance_unchanged': unchanged}))
    if not unchanged:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
