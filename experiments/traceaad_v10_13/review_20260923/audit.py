"""Read-only mechanism audit; real runs and frozen runtimes are never modified.

Run from the repository root:
  .venv/bin/python experiments/traceaad_v10_13/review_20260923/audit.py

Live observations are bounded by each run's checkpoint and are explicitly partial.
The crash reproduction uses a tiny in-process stub in a temporary directory.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from core import Evaluation
from traceaad.v10_13 import TraceAADV1013
from traceaad.v10_13.parsing import FENCE_RE, THINK_BLOCK_RE
from traceaad.v10_13.storage import read_journal


def load_json(path):
    return json.loads(path.read_text())


def rows(path):
    """Ignore an unfinished final append without editing the journal."""
    with path.open('rb') as handle:
        for line in handle:
            if not line.endswith(b'\n'):
                break
            if line.strip():
                yield json.loads(line)


def sha(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def audit_batch(version, batch, *, partial=False):
    base = ROOT / 'experiments' / ('traceaad_' + version) / 'results'
    manifest = load_json(base / f'batch_{batch}.json')
    operators = defaultdict(Counter)
    repeated = Counter()
    failure_counts = Counter()
    duplicate_counts = Counter()
    metadata = []
    length_failures = Counter()
    length_example = None
    for lane in manifest['plan']:
        directory = base / lane['task'] / lane['run_name']
        state = load_json(directory / 'tree_state.json')
        limit = state.get('completed_candidates', state.get('completed_attempts'))
        events = [e for e in rows(directory / 'events.jsonl') if e['candidate_id'] <= limit]
        ids = {e['node_id'] for e in events if e.get('node_id') is not None}
        nodes = [n for n in rows(directory / 'nodes.jsonl') if n['id'] in ids]
        keys = {n['id']: ast.dump(ast.parse(n['code']), include_attributes=False) for n in nodes}
        duplicate_counts['nodes'] += len(nodes)
        duplicate_counts['duplicate_ast_nodes'] += len(keys) - len(set(keys.values()))
        prompts = Counter()
        best = None
        too_long = set()
        for e in events:
            op, score = e['operator'], e.get('fitness')
            if e['status'] != 'ok':
                failure_counts[e.get('reason') or e['status']] += 1
            if (e.get('reason') or '').startswith('idea_length_error'):
                too_long.add(e['candidate_id'])
            selected = e.get('parent_selected', e.get('parent_id') is not None and not e.get('repair_of'))
            if selected and op in ('Refine', 'Tune'):
                prompts[(e['parent_id'], op, e['prompt_hash'])] += 1
            if score is None:
                continue
            if e.get('parent_fitness') is not None:
                c = operators[op]
                c['valid'] += 1
                improved = score > e['parent_fitness'] + 1e-10
                c['above_parent'] += improved
                c['above_frontier'] += best is not None and score > best + 1e-10
                if e.get('donor_fitness') is not None:
                    c['with_donor'] += 1
                    c['donor_above_parent'] += e['donor_fitness'] > e['parent_fitness'] + 1e-10
                    c['above_parent_with_donor'] += improved
                    c['above_both'] += score > max(e['parent_fitness'], e['donor_fitness']) + 1e-10
                key = keys.get(e['node_id'])
                c['copy_parent'] += key is not None and key == keys.get(e['parent_id'])
                c['copy_donor'] += e.get('donor_id') is not None and key == keys.get(e['donor_id'])
            best = score if best is None else max(best, score)
        repeated['normal_refine_tune_calls'] += sum(prompts.values())
        repeated['calls_after_first_identical_context'] += sum(n - 1 for n in prompts.values())
        if partial and too_long:
            for call in rows(directory / 'llm_calls.jsonl'):
                if call['candidate_id'] not in too_long or 'response' not in call:
                    continue
                length_failures['responses'] += 1
                text = THINK_BLOCK_RE.sub('', call['response'])
                fences = list(FENCE_RE.finditer(text))
                if len(fences) == 2:
                    try:
                        ast.parse(text[fences[0].end():fences[1].start()])
                    except (SyntaxError, ValueError):
                        pass
                    else:
                        length_failures['with_syntax_valid_code'] += 1
                if length_example is None:
                    length_example = {'run': lane['run_name'], 'candidate_id': call['candidate_id']}
        summary_path = directory / 'logs/run_summary.json'
        summary = load_json(summary_path) if summary_path.exists() else {}
        metadata.append({
            'run_name': lane['run_name'], 'task': lane['task'],
            'completed_candidates': limit, 'budget_used': state['budget_used'],
            'summary_status': summary.get('status'),
            'manifest_status': lane.get('status'),
            'events_count': len(events), 'events_prefix_sha256': sha(events),
            'nodes_count': len(nodes), 'nodes_prefix_sha256': sha(nodes),
        })
    return {
        'batch': batch, 'status': 'partial' if partial else 'finished',
        'scope': 'within-version diagnostics; pooled events are not independent experimental repeats',
        'frontier_rule': 'strict improvement >1e-10 over all earlier valid nodes in this run',
        'operators': dict(operators), 'repeated_contexts': repeated,
        'duplicate_ast': duplicate_counts, 'failure_reasons': failure_counts,
        'idea_length_failures': length_failures, 'idea_length_example': length_example,
        'runs': metadata,
    }


def heldout_summaries():
    batches = {
        'v10_10': '20260910_v1010_formal',
        'v10_11': '20260915_v1011_generic',
        'v10_12': '20260923_v1012',
        'v11_1': '20260923_v111_manual',
    }
    result = {}
    for version, batch in batches.items():
        tasks = {}
        base = ROOT / 'experiments' / ('traceaad_' + version) / ('results_heldout_' + batch)
        for task in ['tsp_construct', 'cvrp_aco', 'op_aco', 'online_bin_packing', 'vrptw_construct']:
            path = base / task / 'results.json'
            data = load_json(path)
            mapping = next(data[key] for key in ['eval_results_by_size', 'results_by_size',
                                                 'results_by_split', 'eval_results_by_scale'] if key in data)
            tasks[task] = {'source': str(path.relative_to(ROOT)),
                           'splits': {key: value['summary'] for key, value in mapping.items()}}
        result[version] = tasks
    return result


class StubEvaluation(Evaluation):
    def __init__(self):
        super().__init__(template_program='def score(x):\n    pass',
                         task_description='Return a numeric score.', safe_evaluate=False)
        self.calls = 0

    def evaluate_program(self, program_str, callable_func, **kwargs):
        self.calls += 1
        return callable_func(1)


class StubLLM:
    def count_prompt_tokens(self, text):
        return len(text.split())

    def draw_sample_with_details(self, *args, **kwargs):
        return {'content': 'Idea: Return the input.\nCode:\n```python\ndef score(x):\n    return x\n```',
                'finish_reason': 'stop'}


def crash_reproduction():
    with tempfile.TemporaryDirectory(prefix='traceaad_review_') as temporary:
        evaluation = StubEvaluation()
        directory = Path(temporary)
        method = TraceAADV1013(evaluation=evaluation, llm=StubLLM(), run_dir=directory,
                              budget=2, n_roots=1)
        original_record_event = method.storage.record_event

        def crash(record):
            original_record_event(record)
            if record['candidate_id'] == 2:
                raise OSError('injected crash after event append and before checkpoint')

        method.storage.record_event = crash
        try:
            method.run()
        except OSError:
            pass
        resumed = TraceAADV1013(evaluation=evaluation, llm=StubLLM(), run_dir=directory,
                               budget=2, n_roots=1)
        resumed.run()
        events = read_journal(directory / 'events.jsonl')
        return {'stub_calls': evaluation.calls, 'checkpoint_budget': resumed.evaluations_used,
                'candidate_ids': [e['candidate_id'] for e in events],
                'evaluation_ids': [e['evaluation_id'] for e in events]}


def main():
    payload = {'captured_at': datetime.now().astimezone().isoformat(), 'batches': {}}
    for version, batch in [('v10_11', '20260915_v1011_generic'),
                           ('v11_1', '20260920_v111_manual'),
                           ('v10_13', '20260923_v1013')]:
        payload['batches'][version] = audit_batch(version, batch, partial=version == 'v10_13')
    payload['heldout'] = heldout_summaries()
    payload['crash_reproduction'] = crash_reproduction()
    runtime = ROOT / 'experiments/traceaad_v10_13/results/runtime_20260923_v1013'
    payload['frozen_source_matches'] = {
        str(path.relative_to(ROOT)): path.read_bytes() == (runtime / path.relative_to(ROOT)).read_bytes()
        for path in (ROOT / 'traceaad/v10_13').glob('*.py')
    }
    path = Path(__file__).with_name('snapshot.json')
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    for version, data in payload['batches'].items():
        print(version, json.dumps({key: data[key] for key in ['status', 'operators',
              'repeated_contexts', 'idea_length_failures', 'duplicate_ast']}, ensure_ascii=False))
    print('saved', path.relative_to(ROOT))


if __name__ == '__main__':
    main()
