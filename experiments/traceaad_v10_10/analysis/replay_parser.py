"""Replay frozen V10.10 responses through the template-rebuild parser (read-only).

Zero LLM calls: every persisted response of the selected batch is re-parsed
with the current parser and compared with the outcome recorded at generation
time. Never writes inside run directories; reports land under
``results/analysis/`` next to the raw artifacts.

Usage:
    python experiments/traceaad_v10_10/analysis/replay_parser.py \
        [--batch 20260910_v1010_formal] [--results-root experiments/traceaad_v10_10/results]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm4ad.base.code import TextFunctionProgramConverter  # noqa: E402
from traceaad.v10_10 import errors  # noqa: E402

TASK_TEMPLATE_MODULES = {
    'tsp_construct': 'benchmarks.tsp_construct.template',
    'cvrp_aco': 'benchmarks.cvrp_aco.template',
    'op_aco': 'benchmarks.op_aco.template',
    'online_bin_packing': 'benchmarks.online_bin_packing.template',
    'vrptw_construct': 'benchmarks.vrptw_construct.template',
}


def _repo_template_text(task: str) -> str | None:
    try:
        module = importlib.import_module(TASK_TEMPLATE_MODULES[task])
    except ImportError:
        return None
    return module.template_program


def _interface_from_template(program_text: str):
    program = TextFunctionProgramConverter.text_to_program(program_text)
    function = program.functions[0]
    return errors.expected_interface(function.name, function.args)


def template_interface(task: str, run_dir: Path | None = None):
    """The target-function interface a run actually saw.

    Prefers the template program frozen in the run's own checkpoint
    (mechanism.evaluation_config.template_program) over the current repo
    template, and reports whether the two still agree.
    """
    if run_dir is not None:
        state_path = run_dir / 'tree_state.json'
        if state_path.exists():
            program_text = (json.loads(state_path.read_text())
                            .get('mechanism', {}).get('evaluation_config', {})
                            .get('template_program'))
            if program_text:
                frozen = hashlib.sha256(program_text.encode()).hexdigest()
                repo_text = _repo_template_text(task)
                return _interface_from_template(program_text), {
                    'template_program': program_text,
                    'source': 'frozen_checkpoint',
                    'template_sha256': frozen,
                    'repo_template_differs': (
                        None if repo_text is None
                        else hashlib.sha256(repo_text.encode()).hexdigest() != frozen),
                }
    program_text = _repo_template_text(task)
    return _interface_from_template(program_text), {
        'source': 'repo', 'template_program': program_text,
    }


def read_complete_lines(path: Path) -> dict:
    """Complete JSONL records only; a trailing partial write is reported, never parsed.

    A complete line that fails JSON parsing is interior corruption: the journal
    is then untrustworthy for accounting and replay refuses to continue.
    """
    data = path.read_bytes()
    records, partial_bytes, corrupt_lines = [], 0, 0
    for line in data.splitlines(keepends=True):
        if not line.endswith(b'\n'):
            partial_bytes = len(line)
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            corrupt_lines += 1
    if corrupt_lines:
        raise ValueError(f'{path}: {corrupt_lines} complete but unparseable line(s); '
                         'refusing to replay a corrupted journal')
    return {
        'records': records,
        'snapshot': {
            'path': str(path), 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest(),
            'complete_lines': len(records), 'corrupt_lines': corrupt_lines,
            'partial_tail_bytes': partial_bytes,
            'read_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        },
    }


def old_error_kind(error: str | None) -> str:
    text = error or ''
    if text.startswith('Missing Idea'):
        return 'missing_idea'
    if text.startswith('Expected one unambiguous'):
        return 'no_or_multi_block'
    if text.startswith('Output reached the token limit'):
        return 'truncated'
    if text.startswith('SyntaxError') or text.startswith('ValueError'):
        return 'syntax'
    if text.startswith('Expected exactly one target function'):
        return 'interface'
    return 'other'


def new_error_kind(error: str | None) -> str | None:
    text = error or ''
    if not text:
        return None
    if text.startswith('SyntaxError') or text.startswith('ValueError'):
        return 'syntax_or_compile'
    if 'function definition' in text or 'must declare the parameters' in text:
        return 'interface'
    return 'extraction'  # ambiguous block, empty block, truncation, finish reason


def classify(event: dict | None, new_error: str | None) -> str:
    if event is None:
        return 'response_without_event'
    if event['status'] == 'invalid_output':
        if new_error is None:
            return 'recovered_by_new_parser'
        if old_error_kind(event.get('error')) == 'missing_idea' and \
                new_error_kind(new_error) != 'extraction':
            return 'missing_idea_masked_real_code_error'
        return 'still_rejected'
    # ok / eval_failed / duplicate_code all parsed successfully at generation time.
    return 'both_accepted' if new_error is None else 'new_parser_rejects_old_valid'


def replay_run(run_dir: Path, task: str) -> dict:
    interface, interface_info = template_interface(task, run_dir)
    events_read = read_complete_lines(run_dir / 'events.jsonl')
    calls_read = read_complete_lines(run_dir / 'llm_calls.jsonl')
    events = {record['candidate_id']: record for record in events_read['records']}
    rows, code_mismatches = [], []
    classifications = Counter()
    new_sources = Counter()
    for call in calls_read['records']:
        if 'response' not in call:
            classifications['transport_error_call'] += 1
            continue
        parsed, source, new_error = errors.parse_candidate(
            call['response'], call.get('finish_reason') or 'unknown', interface,
            interface_info['template_program'])
        event = events.get(call['candidate_id'])
        outcome = classify(event, new_error)
        classifications[outcome] += 1
        row = {
            'call_id': call['call_id'], 'candidate_id': call['candidate_id'],
            'stage': call.get('stage', 'generation'),
            'repair_of': call.get('repair_of'),
            'old_status': event['status'] if event else None,
            'old_error_kind': old_error_kind(event.get('error'))
            if event and event['status'] == 'invalid_output' else None,
            'new_error': new_error, 'classification': outcome,
        }
        if parsed is not None:
            new_sources[source] += 1
            row['idea_source'] = source
            code_hash = hashlib.sha256(parsed[1].encode()).hexdigest()
            if event is not None and event.get('code_hash') and code_hash != event['code_hash']:
                outcome = 'code_text_differs'
                row['classification'] = outcome
                classifications['both_accepted'] -= 1
                classifications[outcome] += 1
                code_mismatches.append({
                    **row, 'response_excerpt': call['response'][:400],
                    'new_hash': code_hash, 'old_hash': event['code_hash']})
        rows.append(row)
    paired = {row['candidate_id'] for row in rows}
    unpaired_events = sorted(set(events) - paired)
    repairs = [e for e in events_read['records'] if e.get('repair_of')]
    repair_matrix = Counter(
        (events[r['repair_of']]['status'] if r['repair_of'] in events else 'missing_origin',
         r['status']) for r in repairs)
    return {
        'run_dir': str(run_dir),
        'interface': interface_info,
        'run_config_method': json.loads((run_dir / 'run_config.json').read_text())['method']
        if (run_dir / 'run_config.json').exists() else None,
        'event_status': Counter(e['status'] for e in events_read['records']),
        'invalid_by_old_error': Counter(
            old_error_kind(e.get('error')) for e in events_read['records']
            if e['status'] == 'invalid_output'),
        'classifications': classifications,
        'new_idea_sources': new_sources,
        'repair_matrix': {f'{a} -> {b}': n for (a, b), n in sorted(repair_matrix.items())},
        'events_without_response': unpaired_events,
        'code_mismatches': code_mismatches,
        'snapshots': [events_read['snapshot'], calls_read['snapshot']],
        'rows': rows,
    }


def replay_batch(results_root: Path, batch: str) -> dict:
    manifest = json.loads((results_root / f'batch_{batch}.json').read_text())
    plan = manifest['plan']
    runs = []
    for row in plan:
        run_dir = results_root / row['task'] / row['run_name']
        if not run_dir.exists():
            runs.append({'run_dir': str(run_dir), 'missing': True})
            continue
        runs.append(replay_run(run_dir, row['task']))
    report = {
        'batch': batch, 'manifest_runs': len(plan),
        'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'parser_policy': errors.PARSE_POLICY,
        'interface_policy': 'per-run frozen checkpoint template with repo hash cross-check',
        'runs': runs,
    }
    totals = Counter()
    sources = Counter()
    for run in runs:
        totals.update(run.get('classifications', Counter()))
        sources.update(run.get('new_idea_sources', Counter()))
    report['totals'] = dict(totals)
    report['new_idea_sources'] = dict(sources)
    return report


def print_summary(report: dict) -> None:
    print(f"# Parser replay — batch {report['batch']} "
          f"({report['generated_at']}, policy {report['parser_policy']})\n")
    print(f"runs: {report['manifest_runs']}\n")
    print('| classification | calls |')
    print('|---|---|')
    for key, count in sorted(report['totals'].items()):
        print(f'| {key} | {count} |')
    print('\n| new idea source (accepted calls) | count |')
    print('|---|---|')
    for key, count in sorted(report['new_idea_sources'].items()):
        print(f'| {key} | {count} |')
    for run in report['runs']:
        if run.get('missing'):
            print(f"\nMISSING RUN DIR: {run['run_dir']}")
            continue
        print(f"\n## {Path(run['run_dir']).name}")
        print('events:', dict(run['event_status']))
        print('invalid (old):', dict(run['invalid_by_old_error']))
        print('repairs:', run['repair_matrix'])
        if run['events_without_response']:
            print('events without persisted response:', run['events_without_response'])
        if run['code_mismatches']:
            print(f"CODE TEXT DIFFERS on {len(run['code_mismatches'])} calls "
                  '(see JSON rows for excerpts)')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', default='20260910_v1010_formal')
    parser.add_argument('--results-root', type=Path, default=REPO_ROOT /
                        'experiments' / 'traceaad_v10_10' / 'results')
    args = parser.parse_args()
    report = replay_batch(args.results_root, args.batch)
    out_dir = args.results_root / 'analysis'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'parser_replay_{args.batch}.json'
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=dict))
    print_summary(report)
    print(f'\nfull report: {out_path}')


if __name__ == '__main__':
    main()
