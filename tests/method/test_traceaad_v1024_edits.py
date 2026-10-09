"""Editing is one atomic candidate delivery, never an unmetered inner search."""

import pytest

from tests.support import TinyEvaluation, TokenLLM, response
from traceaad.common.delivery import DeliveryError, SourceError
from traceaad.common.state import Facts
from traceaad.v10_24 import Config, TraceAADV1024
from traceaad.v10_24.delivery import apply_edits, parse_candidate
from traceaad.v10_24.diagnostics import development_stats

BASE = 'def score(x):\n    value = 1\n    return value\n'
TEMPLATE = 'def score(x):\n    pass\n'


def block(search, replacement):
    return f'<<<<<<< SEARCH\n{search}\n=======\n{replacement}\n>>>>>>> REPLACE'


def edit(search, replacement):
    return 'Analysis:\nChange: improve score\nEffect: change result\nStatus: continue_request\nDesign: updated computation\nEdits:\n' + block(search, replacement)


def run(path, replies, budget=None):
    return TraceAADV1024(evaluation=TinyEvaluation(), llm=TokenLLM(*replies), run_dir=path,
                         task='tiny', config=Config(budget=budget or len(replies), roots=1, init_attempt_limit=1))


def test_batch_matches_original_and_preserves_untouched_source():
    source = '# keep this exact spacing\ndef score(x):\n    value = 1\n    return value\n'
    payload = block('    value = 1', '    value = 2') + '\n' + block('    return value', '    return value * x')
    updated, count = apply_edits(source, payload)
    assert count == 2
    assert updated == source.replace('value = 1', 'value = 2').replace('return value', 'return value * x')
    with pytest.raises(DeliveryError, match='not found'):
        apply_edits(source, block('value = 1', 'value = 2') + '\n' + block('value = 2', 'value = 3'))


@pytest.mark.parametrize('payload,message', [
    (block('missing', 'x'), 'not found'),
    (block('value = 1', 'value = 2') + '\n' + block('missing', 'x'), 'not found'),
    (block('value', 'other'), 'ambiguous'),
    (block('', 'x'), 'empty SEARCH'),
    (block('    value = 1', '') + '\n' + block('value = 1', 'value = 2'), 'overlapping'),
    ('<<<<<<< SEARCH\nvalue = 1\n=======\nvalue = 2', 'incomplete'),
    (block('value = 1', 'value = 2') + '\n<<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE', 'malformed'),
])
def test_bad_batch_fails_without_partial_application(payload, message):
    with pytest.raises(DeliveryError, match=message):
        apply_edits(BASE, payload)
    assert BASE == 'def score(x):\n    value = 1\n    return value\n'


def test_wrappers_and_delete_are_deterministically_parsed():
    text = 'A proposed change follows.\n' + edit('    value = 1\n', '').replace('Edits:\n', 'Edits:\n```diff\n') + '\n```\n'
    code, idea, delivery = parse_candidate(text, 'stop', TEMPLATE, BASE)
    assert code == 'def score(x):\n    return value\n' and idea == 'updated computation'
    assert delivery['edit_count'] == 1


def test_syntax_failure_retains_full_applied_source_for_repair():
    with pytest.raises(SourceError) as err:
        parse_candidate(edit('return value', 'return ('), 'stop', TEMPLATE, BASE)
    assert 'def score(x)' in err.value.code and 'return (' in err.value.code
    with pytest.raises(DeliveryError, match='incomplete'):
        parse_candidate(edit('value = 1', 'value = 2'), 'length', TEMPLATE, BASE)


def test_full_fallback_and_init_without_edit_base():
    assert parse_candidate(response(7), 'stop', TEMPLATE)[0].endswith('return 7')
    with pytest.raises(DeliveryError, match='initialization'):
        parse_candidate(edit('value = 1', 'value = 2'), 'stop', TEMPLATE)


def test_patch_failure_is_paid_but_not_evaluated_and_next_request_sees_it(tmp_path):
    m = run(tmp_path, [response(1), edit('not present', 'return 2'), edit('return 1', 'return 3')])
    summary = m.run()
    assert summary['budget_used'] == 3 and summary['model_calls'] == 3
    assert m.attempts_table[2]['status'] == 'delivery_failed'
    assert m.attempts_table[2]['entered_evaluation'] is False
    assert 'SEARCH not found' in m.llm.calls[2][0]
    assert 'not present' in m.llm.calls[2][0] and 'Unapplied edit submission' in m.llm.calls[2][0]
    assert m.attempts_table[3]['action'] == 'Refine'
    assert m.programs[3]['code'] == 'def score(x):\n    return 3\n'
    d = development_stats(m.facts)['delivery']['edit']
    assert (d['attempts'], d['evaluated_candidates'], d['model_calls']) == (2, 1, 2)
    assert d['output_tokens'] is None  # missing usage is not invented savings


def test_edit_invalid_source_repairs_materialized_program(tmp_path):
    m = run(tmp_path, [response(1), edit('return 1', 'return ('), edit('return (', 'return 4'), response(5), response(6)])
    m.run()
    assert m.attempts_table[2]['status'] == 'invalid_source'
    assert m.programs[2]['idea'] == 'updated computation'
    assert m.attempts_table[3]['action'] == 'Repair' and m.attempts_table[3]['edit_base_id'] == 2
    assert m.attempts_table[4]['parent_id'] == 3
    assert m.programs[2]['code'] in m.llm.calls[2][0]


def test_edits_preserve_source_and_canonical_deduplication(tmp_path):
    source = 'def helper(x):\n    return "literal"  # untouched\n\ndef score(x):\n    return 1\n'
    full = 'Design: initial\n```python\n' + source + '```'
    m = run(tmp_path, [full, edit('return 1', 'return 2'),
        'Design: formatting only\n```python\ndef helper(x):\n    return \'literal\'\ndef score(x):\n    return 2\n```'])
    m.run()
    assert m.programs[2]['code'] == source.replace('return 1', 'return 2')
    assert m.attempts_table[3]['status'] == 'duplicate'
    assert Facts(tmp_path).programs[2]['code'] == m.programs[2]['code']


def test_edit_base_is_host_bound_and_resume_keeps_applied_state(tmp_path):
    replies = [response(1), edit('return 1', 'return 2'), edit('return 2', 'return 3')]
    m = run(tmp_path, replies[:2], budget=3)
    while m.attempts < 2:
        (m._roots if m.phase == 'roots' else m._search)()
    resumed = run(tmp_path, replies[2:], budget=3)
    resumed.run()
    assert resumed.attempts_table[3]['edit_base_id'] == 2
    assert resumed.programs[3]['fitness'] == -3


def test_conflicting_declared_edit_base_is_not_silently_rebound(tmp_path):
    wrong = edit('return 2', 'return 3').replace('Analysis:\n', 'Analysis:\nBase: 1\n')
    m = run(tmp_path, [response(1), edit('return 1', 'return 2'), wrong, response(4), response(5)])
    m.run()
    assert m.attempts_table[3]['status'] == 'delivery_failed'
    assert 'conflicts' in m.attempts_table[3]['error']
    assert m.attempts_table[4]['parent_id'] == 2


def test_edit_batch_uses_one_evaluation_and_one_budget_attempt(tmp_path):
    initial = 'Design: initial\n```python\n' + BASE + '```'
    edits = edit('value = 1', 'value = 2') + '\n' + block('return value', 'return value + x')
    m = run(tmp_path, [initial, edits])
    summary = m.run()
    assert summary['budget_used'] == summary['model_calls'] == 2
    assert m.attempts_table[2]['delivery']['edit_count'] == 2
    assert m.programs[2]['fitness'] == -3
    assert sum(a['entered_evaluation'] for a in m.attempts_table.values()) == 2


def test_old_full_only_checkpoint_cannot_silently_resume_under_edit_prompts(tmp_path):
    from traceaad.common.storage import read_json, write_json
    m = run(tmp_path, [response(1)], budget=5)
    m._roots()
    checkpoint = read_json(tmp_path / 'resume.json')
    checkpoint['state']['identity'].pop('delivery_protocol')
    write_json(tmp_path / 'resume.json', checkpoint)
    with pytest.raises(ValueError, match='same configuration'):
        run(tmp_path, [], budget=5)


def test_mixed_full_and_edit_payload_is_rejected():
    text = 'Code:\n```python\n' + BASE + '```\n' + edit('value = 1', 'value = 2')
    with pytest.raises(DeliveryError, match='not both'):
        parse_candidate(text, 'stop', TEMPLATE, BASE)
