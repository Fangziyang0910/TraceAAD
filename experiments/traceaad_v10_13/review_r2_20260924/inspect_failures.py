"""Offline response-envelope inspection; no generation or candidate execution.

Recovery below is diagnostic only, not a proposed production acceptance policy.
Multiple distinct payloads are excluded. Passing parsing proves only syntax and
interface acceptance, not fitness or preservation of the author's intended code.
"""
from collections import Counter
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'experiments/traceaad_v10_13/results/runtime_20260924_v1013r2'))
from traceaad.v10_13.parsing import THINK_BLOCK_RE, parse_candidate, response_object, template_target
from traceaad.v10_13.storage import read_journal

HERE = Path(__file__).resolve().parent


def inspect():
    bound = json.loads((HERE / 'progress_0121_protocol.json').read_text())
    results = ROOT / 'experiments/traceaad_v10_13/results'
    decoder = json.JSONDecoder()
    counts, by_task, examples, cases = Counter(), {}, {}, []
    for run in bound['runs']:
        directory = results / run['task'] / run['run_name']
        limit = run['effective_candidate_boundary']
        events = [e for e in read_journal(directory / 'events.jsonl') if e['candidate_id'] <= limit]
        calls = [c for c in read_journal(directory / 'llm_calls.jsonl') if c['candidate_id'] <= limit]
        bycall = {c['call_id']: c for c in calls}
        state = json.loads((directory / 'tree_state.json').read_text())
        nodes = {n['id']: n for n in state['nodes']}
        template = importlib.import_module('benchmarks.' + run['task'] + '.template').template_program
        interface, _ = template_target(template)
        task_counts = by_task.setdefault(run['task'], Counter())
        for event in events:
            if event['status'] != 'invalid_output':
                continue
            call = bycall[f"{event['candidate_id']}:{event['llm_calls']}"]
            raw = THINK_BLOCK_RE.sub('', call['response']).strip()
            objects = {}
            if response_object(raw) is None:
                # Inspect embedded JSON only; never repair escaping or invent text.
                for index, char in enumerate(raw):
                    if char != '{':
                        continue
                    try:
                        value, end = decoder.raw_decode(raw[index:])
                    except ValueError:
                        continue
                    if isinstance(value, dict) and value.get('mode') in ('full', 'edit', 'context'):
                        key = json.dumps(value, sort_keys=True, ensure_ascii=False)
                        objects[key] = (value, index, index + end)
            tag = 'unrecovered_or_already_valid_json'
            parse_error = None
            if len(objects) == 1:
                payload, start, end = next(iter(objects.values()))
                if payload['mode'] == 'context':
                    tag = 'embedded_context_request'
                else:
                    parent = nodes.get(event.get('parent_id'))
                    parsed, parse_error = parse_candidate(
                        json.dumps(payload), call['finish_reason'], interface, template,
                        base_code=parent['code'] if parent else None,
                        allowed_donor_ids=event.get('reference_ids', []))
                    tag = 'embedded_proposal_parses' if parsed else 'embedded_proposal_still_invalid'
                case = {'run': run['run_name'], 'candidate_id': event['candidate_id'],
                        'original_reason': event.get('reason'), 'tag': tag,
                        'finish_reason': call.get('finish_reason'), 'mode': payload['mode'],
                        'prefix': raw[:start][:200], 'suffix': raw[end:][-120:],
                        'parse_error': parse_error}
                cases.append(case)
                examples.setdefault(tag, case)
            elif len(objects) > 1:
                tag = 'multiple_distinct_embedded_payloads'
            counts[tag] += 1
            task_counts[tag] += 1
            counts['finish_' + str(call.get('finish_reason'))] += 1
    return {'status': 'partial', 'boundary_source': 'progress_0121_protocol.json',
            'scope': 'offline syntax and envelope inspection, no evaluator calls',
            'counts': dict(counts),
            'by_task': {k: dict(v) for k, v in by_task.items()},
            'examples': examples, 'cases': cases}


if __name__ == '__main__':
    result = inspect()
    target = HERE / 'progress_0121_envelopes.json'
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, ensure_ascii=False, indent=2))
