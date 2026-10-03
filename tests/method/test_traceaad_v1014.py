"""V10.14: delivery fidelity and opportunity/context invariants."""

from dataclasses import replace

import pytest

from tests.support import TinyEvaluation, TokenLLM, text_candidate as payload
from traceaad.v10_14 import Config, TraceAADV1014
from traceaad.v10_14.edits import parse_response
from traceaad.v10_14.selection import opportunity_key, select_parent, syntax_id, unique_archive



def method(path, *responses, **settings):
    cfg = Config(**{**dict(budget=8, max_evaluations=8, init_proposals=1), **settings})
    return TraceAADV1014(evaluation=TinyEvaluation(), llm=TokenLLM(*responses), run_dir=path, config=cfg)


@pytest.mark.parametrize('label', ['Idea:', '**Idea:**', '**Idea**:', '### Idea', '思路：', '算法思路:'])
@pytest.mark.parametrize('position', ['before', 'after'])
def test_multiline_idea_before_or_after_complete_module(label, position):
    code = 'import math\nOFFSET = 2\nclass Helper:\n    value = 1\ndef score(x):\n    return math.sqrt(x) + OFFSET + Helper.value\n'
    idea = '多行机制\n改变决策条件，保留反例。'
    block = f'```python\n{code}\n```'
    explanation = f'{label}\n{idea}'
    raw = f'{explanation}\ncode:\n{block}' if position == 'before' else f'{block}\n{explanation}'
    metadata = {}
    parsed, rationale = parse_response(raw, 'stop', TinyEvaluation().template_program, metadata=metadata)
    assert parsed == code and rationale == idea
    assert metadata['idea_status'] == 'present'
    assert metadata['idea_sources'] == [{'position': position, 'kind': 'explicit_label'}]


def test_unlabelled_prose_is_preserved_and_missing_idea_does_not_reject_code():
    block = '```python\ndef score(x):\n    return "Idea: code is not explanation"\n```'
    for raw, expected in [(block, ''), (f'Final implementation:\n{block}', ''),
                          (f'Use the input as the baseline.\n{block}', 'Use the input as the baseline.'),
                          (f'{block}\nThis changes the decision boundary.', 'This changes the decision boundary.')]:
        metadata = {}
        code, idea = parse_response(raw, 'stop', TinyEvaluation().template_program, metadata=metadata)
        assert idea == expected
        assert 'Idea: code' in code
        assert metadata['idea_status'] == ('present' if expected else 'missing_in_response')
        if expected:
            assert metadata['idea_sources'][0]['kind'] == 'unlabelled_prose'


def test_multiple_blocks_do_not_erase_response_explanations():
    first = '```python\ndef score(x): return 1\n```'
    second = '```python\ndef score(x): return 2\n```'
    code, idea = parse_response(f'Idea: First draft\n{first}\nFinal implementation:\n{second}',
                               'stop', TinyEvaluation().template_program)
    assert 'return 2' in code and idea == 'First draft'
    code, idea = parse_response(f'Idea: Final choice\nFinal implementation:\n{first}\nIdea: Other draft\n{second}',
                               'stop', TinyEvaluation().template_program)
    assert 'return 1' in code and idea == 'Final choice\n\nOther draft'


def test_explicit_idea_precedes_prose_fallback_and_empty_label_does_not_hide_prose():
    block = '```python\ndef score(x): return 1\n```'
    _, idea = parse_response(f'Here is the implementation.\n{block}\nIdea: Use a constant.',
                             'stop', TinyEvaluation().template_program)
    assert idea == 'Use a constant.'
    metadata = {}
    _, idea = parse_response(f'Use a constant.\nIdea:\ncode:\n{block}',
                             'stop', TinyEvaluation().template_program, metadata=metadata)
    assert idea == 'Use a constant.'
    assert metadata['idea_sources'][0]['kind'] == 'unlabelled_prose'


def test_truncation_never_salvages_partial_code():
    with pytest.raises(ValueError, match='incomplete'):
        parse_response(payload(), 'length', TinyEvaluation().template_program)


def test_indented_illustration_does_not_swallow_final_code_or_idea():
    raw = ('Idea:\nChoose the smaller score.\n'
           '    ```python\n    score = distance + penalty\n    ```\n'
           'Idea: Return the selected value.\n\ncode:\n'
           '```python\ndef score(x):\n    return x + 1\n```')
    code, idea = parse_response(raw, 'stop', TinyEvaluation().template_program)
    assert code == 'def score(x):\n    return x + 1'
    assert idea == 'Choose the smaller score.\n\nReturn the selected value.'


def test_invalid_code_envelope_still_preserves_supplied_idea(tmp_path):
    raw = 'Idea: Use the input.\ncode:\n```python\ndef score(x): return x\n```\n```'
    m = method(tmp_path, raw, budget=2, init_proposals=1)
    m._initialize()
    attempt = m.facts.tables['attempt'][1]
    assert attempt['status'] == 'delivery_failed'
    assert attempt['idea'] == 'Use the input.'
    assert m.ledger.candidates == m.ledger.calls == 1
    assert m.ledger.evaluations == 0


def test_paid_delivery_failure_and_syntax_failure_preserve_idea(tmp_path):
    m = method(tmp_path, '```python\ndef score(x): return 1',
               payload(idea='Retained explanation', code='def score(x)\n return 1'),
               budget=4, init_proposals=2)
    m._initialize()
    m._initialize()
    assert m.ledger.candidates == m.ledger.calls == 2
    assert m.ledger.evaluations == 0
    assert all(a['status'] == 'delivery_failed' for a in m.facts.tables['attempt'].values())
    assert m.facts.tables['attempt'][2]['idea'] == 'Retained explanation'
    assert len(m.llm.calls) == 2
    assert all('response_format' not in call[1] for call in m.llm.calls)


def test_rank_selection_is_unit_invariant_and_tie_fair():
    anchors = {i: {'id': i, 'artifact_id': str(i), 'fitness': q}
               for i, q in enumerate([-800, -744.5, -744.5, -729], 1)}
    counts = {'1': 0, '2': 4, '3': 0, '4': 9}
    parent, info = select_parent(anchors, counts)
    for scale, offset in [(100, 700), (.001, -17), (1, 200000)]:
        transformed = {i: {**a, 'fitness': a['fitness']*scale+offset} for i, a in anchors.items()}
        selected, trace = select_parent(transformed, counts)
        assert selected['id'] == parent['id']
        assert trace['selection_quality'] == info['selection_quality']
    tied = {i: {**a, 'fitness': 10} for i, a in anchors.items()}
    assert select_parent(tied, {'1': 1})[0]['id'] == 2
    assert select_parent(tied, {})[1]['selection_quality'] == .5
    # A >1-unit fitness deficit no longer creates a permanent raw-unit barrier.
    two = {1: anchors[1], 4: anchors[4]}
    assert select_parent(two, {'4': 100})[0]['id'] == 1
    assert select_parent(two, {'4': 100}, policy='raw_count')[0]['id'] == 4


def test_syntax_accounts_do_not_reset_for_comments_or_formatting():
    a = 'def score(x):\n    return x + 1\n'
    b = '# different story\ndef score( x ):\n    return (x+1) # same computation\n'
    assert syntax_id(a) == syntax_id(b)
    for c in ['def score(x):\n    return x+2', 'def score(x):\n    "docstring"\n    return x+1']:
        assert syntax_id(a) != syntax_id(c)
    anchors = {i: {'id': i, 'artifact_id': str(i), 'fitness': 1, 'syntax_id': syntax_id(c)}
               for i, c in [(1, a), (2, b), (3, 'def score(x):\n return x+2')]}
    assert len(unique_archive(anchors)) == 2
    assert select_parent(anchors, {syntax_id(a): 20})[0]['id'] == 3


def test_independent_pivot_has_no_parent_information_and_no_parent_charge(tmp_path):
    m = method(tmp_path, payload(idea='PARENT_RATIONALE_CANARY', code='# PARENT_CODE_CANARY\ndef score(x):\n return 1'),
               payload(2), payload(3))
    m._initialize()
    m._initialize()
    m._initialize()
    assert m.phase == 'search'
    m._contract = lambda anchor: ('Pivot', 'None', None)
    counts = dict(m.parent_counts)
    m._schedule_search()
    m._step_session()
    request = m.facts.tables['request'][3]
    prompt = request['prompt']
    assert request['context_mode'] == 'independent'
    assert request['evidence_ids'] == [] and request['donor_id'] is None
    for forbidden in ['PARENT_', '# Current program', '# Search feedback', '# Limited training-probe', '# Active, revisable']:
        assert forbidden not in prompt
    attempt = m.facts.tables['attempt'][3]
    assert attempt['parent_id'] is None and attempt['executed_action'] == 'Pivot'
    assert not attempt['selection_applied_to_parent']
    assert m.anchors[3]['parent_id'] is None and m.anchors[3]['revision_id'] is None
    assert m.parent_counts == counts
    assert m.ledger.candidates == 3 and m.ledger.channel_used['main'] == 1
    m._finish_session()
    restored = method(tmp_path)
    assert restored.parent_counts == m.parent_counts
    assert restored.rng.getstate() == m.rng.getstate()


def test_anchored_control_keeps_parent_and_plain_delivery(tmp_path):
    m = method(tmp_path, payload(), payload(2), payload(3), pivot_context='anchored')
    for _ in range(3):
        m._initialize()
    m._contract = lambda anchor: ('Pivot', 'None', None)
    m._schedule_search()
    m._step_session()
    request = m.facts.tables['request'][3]
    assert request['context_mode'] == 'anchored' and '# Current program' in request['prompt']
    assert m.facts.tables['attempt'][3]['parent_id'] is not None
    assert request['output_mode'] == 'full' and not request.get('response_format')


def test_exact_syntax_reuse_shares_count_but_retains_facts_and_repair(tmp_path):
    m = method(tmp_path, payload(code='def score(x):\n return 1'), payload(code='# alias\ndef score(x):\n return 1'),
               payload(idea='Test unresolved dependency', code='def score(x):\n return missing'), payload(2),
               pivot_context='anchored')
    for _ in range(3):
        m._initialize()
    assert len(m.anchors) == 2 and len(unique_archive(m.anchors)) == 1
    m._contract = lambda anchor: ('Refine', 'None', None)
    m._schedule_search()
    m._step_session()
    m._finish_session()
    assert m.repairs
    m._schedule_search()
    m._step_session()
    attempt = m.facts.tables['attempt'][4]
    assert attempt['executed_action'] == 'Repair' and attempt['repair_of'] == 3
    assert sum(m.parent_counts.values()) == 3
    assert m.parent_counts[opportunity_key(m.anchors[1])] == 3
    summary = m.prompts.local_summary(m.anchors[2])
    assert summary['Refine']['tied'] == 1 and summary['Refine']['failed'] == 1
    prompt = m.prompts.build(m.anchors[2])['prompt']
    assert 'same_syntax_paid_attempts' in prompt
    m.prompts.config = replace(m.config, evidence_policy='none')
    assert '# Search feedback' not in m.prompts.build(m.anchors[2])['prompt']


def test_idea_is_preserved_in_anchor_revision_and_history(tmp_path):
    idea = ('mechanism ' * 700).strip()
    m = method(tmp_path, payload(), payload(2, idea=idea))
    m._initialize()
    m._initialize()
    for table in ('anchor', 'attempt', 'revision'):
        fact = m.facts.tables[table][2]
        assert fact['idea'] == idea
        assert fact['idea_metadata']['display_tokens'] <= 500
        assert fact['idea_metadata']['truncated']
    prompt = m.prompts.build(m.anchors[2])
    assert 'revision:2' in prompt['evidence_ids']
    assert 'mechanism' in prompt['prompt']


def test_finalist_slots_are_not_spent_on_same_syntax(tmp_path):
    m = method(tmp_path, payload(1), payload(code='# new description\ndef score(x):\n return 1\n'),
               payload(2), budget=3)
    m._contract = lambda a: ('Refine', 'None', None)
    m.run()
    assert len(m.anchors) == 3
    assert len(m.finalists) == 2
    assert m.anchors[m.finalists[0]]['fitness'] == 2
