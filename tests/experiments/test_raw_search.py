import copy
import hashlib
import json

import pytest

from experiments.infra.migrations import raw_search
from traceaad.v12_1.storage import append_jsonl, read_json, rows, write_json


def fixture(tmp_path):
    code = 'def solve(x): return x'
    key = hashlib.sha256(code.encode()).hexdigest()
    refs = [{'reference': 10.}, {'reference': 100.}]
    event = {'kind': 'candidate', 'candidate_id': 1, 'valid': True, 'fitness': -25.,
             'program': {'id': 1, 'key': key, 'fitness': -25., 'score': -25.}}
    record = {'key': key, 'valid': True, 'score': -25., 'protocol': 'old',
              'timeout_seconds': 30, 'function_seconds_limit': 10}
    profile = {'kind': 'method', 'profiler_sample_order': 1, 'data': {
        'event': 'evaluation_profile', 'score': -25., 'instance_scores': [50., 0.]}}
    return code, key, refs, event, record, profile


def test_plan_preserves_instance_weights_and_historical_limits(tmp_path):
    _, _, refs, event, record, profile = fixture(tmp_path)
    candidates, jobs, exact = raw_search.plan([event, profile], [record], refs)
    identity = candidates[1]
    assert exact[identity]['fitness'] == 52.5
    assert raw_search.execution(jobs[identity], {'method_params': {
        'function_budget_scope': 'mean_instance', 'eval_timeout_seconds': 120}}) == {
        'seeds': [730241], 'timeout_seconds': 30, 'function_seconds': 10,
        'function_budget_scope': 'instance'}
    assert not raw_search.plan([event], [record], refs)[2]
    with pytest.raises(ValueError, match='does not match'):
        raw_search.plan([event, {**profile, 'data': {**profile['data'], 'score': -24.}}], [record], refs)


def test_same_program_across_protocol_change_gets_distinct_measurements(tmp_path):
    _, _, refs, event, record, _ = fixture(tmp_path)
    second = {**record, 'protocol': 'new', 'timeout_seconds': 120, 'function_budget_scope': 'mean_instance'}
    candidates, jobs, exact = raw_search.plan([event, {**event, 'candidate_id': 2}], [record, second], refs)
    assert len(jobs) == 2 and candidates[1] != candidates[2]
    assert not exact


def test_publish_keeps_decisions_and_counts_but_replaces_reported_scores(tmp_path):
    _, key, refs, event, record, profile = fixture(tmp_path)
    backup = tmp_path / 'logs/raw_search_originals'
    append_jsonl(backup / 'evaluations.jsonl', record)
    write_json(tmp_path / 'resume.json', {'files': {'events.jsonl': 3}, 'method_state': {'q_max': 25.}})
    event['attempt'] = {'function_key': 'function-only-hash'}
    events = [copy.deepcopy(event), profile, {'kind': 'population', 'members': [
        {'function_key': 'function-only-hash', 'score': -25.}]}]
    candidates, jobs, exact = raw_search.plan(events, [record], refs)
    raw_search.publish(tmp_path, backup, events, candidates, exact, refs)
    rewritten = list(rows(tmp_path / 'events.jsonl'))
    assert rewritten[0]['fitness'] == rewritten[0]['program']['score'] == 52.5
    assert rewritten[0]['valid'] is True
    assert rewritten[1]['data']['historical_score'] == -25.
    assert 'score' not in rewritten[1]['data']
    assert rewritten[2]['members'][0]['score'] == 52.5
    assert list(rows(backup / 'evaluations.jsonl'))[0]['score'] == -25.
    assert list(rows(tmp_path / 'evaluations.jsonl'))[0]['score'] == 52.5
    assert read_json(tmp_path / 'resume.json')['files']['events.jsonl'] == (tmp_path / 'events.jsonl').stat().st_size


def test_publish_preserves_failed_legacy_recovery_profiles(tmp_path):
    _, key, refs, event, record, profile = fixture(tmp_path)
    backup = tmp_path / 'logs/raw_search_originals'
    append_jsonl(backup / 'evaluations.jsonl', record)
    append_jsonl(backup / 'recovery_profiles.jsonl', {'key': key, 'score': -25., 'instance_scores': [-50., 0.]})
    append_jsonl(backup / 'recovery_profiles.jsonl', {'key': 'failed', 'score': None, 'instance_scores': []})
    candidates, jobs, exact = raw_search.plan([event, profile], [record], refs)
    raw_search.publish(tmp_path, backup, [event, profile], candidates, exact, refs)
    profiles = list(rows(tmp_path / 'recovery_profiles.jsonl'))
    assert profiles[0]['score'] == 52.5 and profiles[0]['instance_scores'] == [5, 100]
    assert profiles[1] == {'key': 'failed', 'score': None, 'instance_scores': [], 'score_metric': 'raw_objective'}
    assert list(rows(backup / 'recovery_profiles.jsonl'))[1]['score'] is None


@pytest.mark.parametrize('has_program', [True, False])
def test_resume_does_not_repeat_evaluation_and_failed_backfill_stays_missing(tmp_path, monkeypatch, has_program):
    code, key, refs, event, record, profile = fixture(tmp_path)
    write_json(tmp_path / 'run_config.json', {'task': 'graph_colouring', 'method_params': {}})
    write_json(tmp_path / 'summary.json', {'status': 'finished'})
    write_json(tmp_path / 'exact_objective_conversion.json', {'summary': {'reference_values': [10., 100.]}})
    append_jsonl(tmp_path / 'events.jsonl', event)
    append_jsonl(tmp_path / 'evaluations.jsonl', record)
    if has_program:
        append_jsonl(tmp_path / 'programs.jsonl', {'key': key, 'code': code})
    (tmp_path / 'raw_search_evaluations.jsonl').write_text('{"identity":')
    monkeypatch.setitem(raw_search.CLASSES, 'graph_colouring', lambda **kw: type('Task', (), {'_rows': refs})())
    monkeypatch.setattr(raw_search, 'convert_final_results', lambda p: None)
    calls = []

    class Evaluator:
        def __init__(self, *a, **k): pass
        def evaluate(self, code, key):
            calls.append(key)
            return {'fitness': None, 'failure': {'kind': 'timeout'}, 'evaluations': []}

    monkeypatch.setattr(raw_search, 'InstanceProgramEvaluator', Evaluator)
    first = raw_search.run(tmp_path, 1, '/unused')
    assert first['status'] == 'complete_with_missing' and first['missing_candidates'] == 1
    assert list(rows(tmp_path / 'events.jsonl'))[0]['fitness'] is None
    assert raw_search.run(tmp_path, 1, '/unused') == first
    assert len(calls) == (2 if has_program else 0)
    assert list((tmp_path / 'logs/raw_search_originals').glob('*.partial'))


def test_final_conversion_does_not_convert_already_raw_heldout_twice(tmp_path, monkeypatch):
    from experiments.infra import raw_objectives
    _, key, refs, event, record, profile = fixture(tmp_path)
    write_json(tmp_path / 'run_config.json', {'task': 'graph_colouring'})
    write_json(tmp_path / 'selection.json', {'selected_node': 1, 'selected_key': key,
        'results': [{'node_id': 1, 'fitness': -25.}]})
    write_json(tmp_path / 'summary.json', {'best': {'score': -25., 'fitness': -25.}})
    heldout = [{'split': 'test_300', 'fitness': 42., 'score_metric': 'raw_objective',
                'evaluations': [{'valid': True, 'score': 42.}]}]
    write_json(tmp_path / 'heldout.json', heldout)
    record['instances'] = [{'instance_index': 0, 'valid': True, 'score': -50.},
                           {'instance_index': 1, 'valid': True, 'score': 0.}]
    append_jsonl(tmp_path / 'events.jsonl', {'progress': {'phase': 'freeze'}, 'evaluations': [record]})
    monkeypatch.setitem(raw_objectives.CLASSES, 'graph_colouring', lambda **kw: type('Task', (), {'_rows': refs})())
    result = raw_objectives.convert_final_results(tmp_path)
    assert result['training'] == 52.5 and result['tests'] == {'test_300': 42.}
    assert read_json(tmp_path / 'heldout.json') == heldout


@pytest.mark.parametrize('cached_timeout', [False, True])
def test_timeout_relaxes_budget_and_recovers_previous_missing_score(tmp_path, monkeypatch, cached_timeout):
    code, key, refs, event, record, _ = fixture(tmp_path)
    write_json(tmp_path / 'run_config.json', {'task': 'graph_colouring', 'method_params': {}})
    write_json(tmp_path / 'summary.json', {'status': 'finished'})
    write_json(tmp_path / 'exact_objective_conversion.json', {'summary': {'reference_values': [10., 100.]}})
    append_jsonl(tmp_path / 'events.jsonl', event)
    append_jsonl(tmp_path / 'evaluations.jsonl', record)
    append_jsonl(tmp_path / 'programs.jsonl', {'key': key, 'code': code})
    failed = {'fitness': None, 'failure': {'kind': 'timeout'}, 'evaluations': []}
    if cached_timeout:
        write_json(tmp_path / 'raw_search.json', {'status': 'complete_with_missing'})
        append_jsonl(tmp_path / 'raw_search_evaluations.jsonl', {
            **failed, 'identity': raw_search.signature(record), 'key': key, 'source': 'cpu_reevaluation'})
    monkeypatch.setitem(raw_search.CLASSES, 'graph_colouring', lambda **kw: type('Task', (), {'_rows': refs})())
    monkeypatch.setattr(raw_search, 'convert_final_results', lambda p: None)
    calls = []

    class Evaluator:
        def __init__(self, *a, **kw): self.options = kw
        def evaluate(self, code, key):
            calls.append(self.options)
            if self.options['function_seconds'] is not None:
                return failed
            return {'fitness': 52.5, 'failure': None, 'evaluations': []}

    monkeypatch.setattr(raw_search, 'InstanceProgramEvaluator', Evaluator)
    result = raw_search.run(tmp_path, 1, '/unused')
    assert result['status'] == 'complete' and result['missing_candidates'] == 0
    assert len(calls) == (1 if cached_timeout else 2)
    assert calls[-1]['function_seconds'] is None and calls[-1]['timeout_seconds'] == 600
    stored = list(rows(tmp_path / 'raw_search_evaluations.jsonl'))[-1]
    assert stored['source'] == 'cpu_reevaluation_relaxed'
    assert stored['previous_attempt']['failure']['kind'] == 'timeout'
    assert list(rows(tmp_path / 'events.jsonl'))[0]['fitness'] == 52.5
