"""Replay frozen V10.14 responses through both delivery paths; no generation.

This measures engineering acceptance/information preservation only. It never
evaluates newly recovered code or writes it back into a historical run.
"""

import argparse
import ast
from collections import Counter
import json
from pathlib import Path

from experiments.infra.base import build_task
from traceaad.v10_14.edits import parse_response as old_parse
from traceaad.v10_14_2.edits import parse_response as new_parse


def definitions(code):
    return sorted(n.name for n in ast.parse(code).body
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))


def replay(snapshot, output):
    counts, examples = Counter(), []
    for key in ('acceptance_regression', 'previously_retained_definition_lost', 'calls_without_completed_attempt'):
        counts[key] = 0
    templates = {}
    for path in sorted(Path(snapshot).glob('*_r*_facts.json')):
        task = path.name.rsplit('_r', 1)[0]
        if task not in templates:
            templates[task] = build_task(task, 1)[0].template_program
        template = templates[task]
        facts = json.loads(path.read_text())
        for call in facts['calls'].values():
            if str(call['candidate_id']) not in facts['tables']['attempt']:
                counts['calls_without_completed_attempt'] += 1
            if call.get('error'):
                counts['service_error_not_parsed'] += 1
                continue
            counts['responses'] += 1
            results = {}
            info = {}
            for name, parser in [('old', old_parse), ('new', new_parse)]:
                try:
                    kwargs = {'metadata': info} if name == 'new' else {}
                    code, idea = parser(call['response'], call.get('finish_reason'), template, **kwargs)
                    results[name] = {'code': code, 'idea': idea}
                    counts[f'{name}_accepted'] += 1
                except (SyntaxError, ValueError, TypeError) as exc:
                    results[name] = {'error': str(exc)}
                    counts[f'{name}_rejected'] += 1
            old, new = results['old'], results['new']
            if 'code' in new:
                counts['strategy:' + info['strategy']] += 1
                if 'code' in old:
                    if old['idea'] != new['idea']:
                        counts['idea_changed'] += 1
                    if not old['idea'] and new['idea']:
                        counts['idea_recovered_from_empty'] += 1
                    if set(definitions(old['code'])) - set(definitions(new['code'])):
                        counts['previously_retained_definition_lost'] += 1
                else:
                    counts['newly_accepted_not_rescored'] += 1
                    examples.append({'run': path.stem, 'candidate': call['candidate_id'],
                                     'old_error': old['error'], 'delivery': info})
            elif 'code' in old:
                counts['acceptance_regression'] += 1
                examples.append({'run': path.stem, 'candidate': call['candidate_id'], 'new_error': new['error']})
        print(path.stem, dict(counts), flush=True)
    report = {'counts': dict(counts), 'examples': examples,
              'input_snapshot': str(snapshot),
              'token_measurement': 'Not measured in this offline replay; production and dedicated tests use actual tokenizers.',
              'scope': 'No generation, no rescoring, no historical result mutation; parser acceptance is not execution validity.'}
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    replay(args.snapshot, args.output)


if __name__ == '__main__':
    main()
