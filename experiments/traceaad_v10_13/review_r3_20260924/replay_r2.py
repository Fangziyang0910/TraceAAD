"""Replay recorded r2 responses through r3 parsing without generation/evaluation."""
import importlib
import json
from collections import Counter
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from traceaad.v10_13.parsing import parse_candidate, template_target
from traceaad.v10_13.storage import read_journal
HERE = Path(__file__).resolve().parent
old = ROOT / 'experiments/traceaad_v10_13/review_r2_20260924'
bound = json.loads((old / 'progress_0121_protocol.json').read_text())
counts, examples = Counter(), []
for run in bound['runs']:
    directory = ROOT / 'experiments/traceaad_v10_13/results' / run['task'] / run['run_name']
    limit = run['effective_candidate_boundary']
    events = [e for e in read_journal(directory / 'events.jsonl') if e['candidate_id'] <= limit]
    calls = [c for c in read_journal(directory / 'llm_calls.jsonl') if c['candidate_id'] <= limit]
    bycall = {c['call_id']: c for c in calls}
    state = json.loads((directory / 'tree_state.json').read_text())
    nodes = {n['id']: n for n in state['nodes']}
    template = importlib.import_module('benchmarks.' + run['task'] + '.template').template_program
    interface, _ = template_target(template)
    for event in events:
        call = bycall[f"{event['candidate_id']}:{event['llm_calls']}"]
        parent = nodes.get(event.get('parent_id'))
        parsed, error = parse_candidate(call['response'], call['finish_reason'], interface, template,
                                       base_code=parent['code'] if parent else None,
                                       allowed_donor_ids=event.get('reference_ids', []))
        if event['status'] == 'ok':
            assert parsed is not None and parsed.program_code == nodes[event['node_id']]['code'], (run['run_name'], event['candidate_id'], error)
            counts['existing_valid_code_preserved'] += 1
        elif event['status'] == 'invalid_output':
            counts['old_invalid_recovered' if parsed else 'old_invalid_still_rejected'] += 1
            if parsed:
                examples.append({'run': run['run_name'], 'candidate_id': event['candidate_id'], 'mode': parsed.mode})
            elif error.startswith('context_error'):
                counts['removed_context_requests_rejected'] += 1
result = {'source_boundary':str(old.relative_to(ROOT) / 'progress_0121_protocol.json'),
          'scope':'offline syntax replay only; no candidate evaluation', 'counts':dict(counts), 'recovered':examples}
target=HERE/'r2_response_replay.json'
assert not target.exists()
target.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(result['counts']))
