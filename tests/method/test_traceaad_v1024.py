"""Development state, resource competition, context boundaries and recovery."""

import json

import pytest

from benchmarks.tasks import CO_TASKS, training_task
from tests.support import TinyEvaluation, TokenLLM, response
from traceaad.common.state import Facts
from traceaad.common.storage import committed_rows, read_json, write_json
from traceaad.v10_24 import Config, TraceAADV1024
from traceaad.v10_24.policy import declarations, frontier
from traceaad.v10_24.prompts import PromptBuilder


def answer(value, base='', status='continue_request', plan=False):
    extra = ('Plan: two blocks\nStage 1: build shared estimates\nStage 2: connect estimates\n'
             'Dependency: output needs the new estimates\nLocation: score return\n') if plan else ''
    return (f'Analysis:\nBase: {base}\nChange: score return\nEffect: change final choice\n'
            f'Evidence: inspect supplied code\nStatus: {status}\n{extra}' + response(value))


def method(path, answers, **config):
    return TraceAADV1024(evaluation=TinyEvaluation(), llm=TokenLLM(*answers), run_dir=path,
                        config=Config(roots=1, init_attempt_limit=1, **config), task='tiny')


def until(m, count):
    while m.attempts < count:
        getattr(m, '_roots' if m.phase == 'roots' else '_search')()


def test_question_is_not_the_first_edit_instruction(tmp_path):
    first = answer(9).replace('Change: score return',
        'Question: Can flatter priors improve sampling?\nChange: replace exponent 2 with 1')
    m = method(tmp_path, [response(10), first, answer(10), answer(11)], budget=4)
    m.run()
    assert m.policy['units'][0]['question'] == 'Can flatter priors improve sampling?'
    assert '"question": "Can flatter priors improve sampling?"' in m.llm.calls[2][0]
    assert '"question": "replace exponent' not in m.llm.calls[2][0]


def test_question_survives_failed_delivery_and_all_trial_outcomes_are_visible(tmp_path):
    first = 'Analysis:\nQuestion: Can flatter priors help?\nChange: test exponent\nno code'
    m = method(tmp_path, [response(10), first, answer(9), answer(10), answer(11)], budget=5)
    m.run()
    assert m.policy['units'][0]['question'] == 'Can flatter priors help?'
    prompt = m.llm.calls[-1][0]
    for i in (2, 3, 4):
        assert f'Trial {i}:' in prompt
    assert 'status=duplicate' in prompt and 'code=1' in prompt
    assert '[Delivery target] Edit code 1' in prompt


def test_worktip_can_regress_without_overwriting_champion_and_freeze(tmp_path):
    m = method(tmp_path, [response(100), answer(90), answer(80), answer(110), answer(105)], budget=5)
    summary = m.run()
    u = m.policy['units'][0]
    assert (u['anchor_id'], u['proposal_id'], u['champion_id'], u['worktip_id']) == (1, 2, 4, 5)
    assert m.attempts_table[4]['parent_id'] == 3
    assert summary['budget_used'] == 5 and summary['best']['id'] == 4
    assert summary['development']['frontier_gain'] == u['anchor_delta'] == 10
    assert len(m.facts.evaluations) == 10


def test_failed_proposal_and_repairs_survive_as_diffs_not_editable_bases(tmp_path):
    bad = 'Design: failed computation\n```python\ndef score(x):\n    return missing_value\n```'
    m = method(tmp_path, [response(100), bad, answer(90), answer(110, base='1'), answer(105)], budget=5)
    m.run()
    u = m.policy['units'][0]
    assert u['proposal_id'] == 2 and not m.programs[2]['valid']
    assert m.attempts_table[3]['action'] == 'Repair' and m.attempts_table[3]['repair_of'] == 2
    assert m.attempts_table[4]['parent_id'] == 3  # anchor was only a diff, not a complete base
    assert m.attempts_table[4]['base_fallback_reason'] == 'missing_or_unavailable_base'
    prompt = m.llm.calls[-1][0]
    assert 'missing_value' in prompt and 'repair_of=2' in prompt
    assert all(f'Trial {i}:' in prompt for i in (2, 3, 4))
    assert u['pending_failure'] is None and u['worktip_id'] == 5


def test_valid_cached_rollback_changes_worktip_without_evaluation(tmp_path):
    m = method(tmp_path, [response(10), answer(20), answer(10), answer(11)], budget=4)
    m.run()
    assert m.attempts_table[3]['status'] == 'duplicate'
    assert m.attempts_table[3]['entered_evaluation'] is False
    assert m.attempts_table[4]['parent_id'] == 1
    assert m.policy['units'][0]['champion_id'] == 2
    assert m.policy['blocks'][0]['spent'] == 3


def test_known_failure_remains_repair_target_and_delivery_failure_does_not_clear_it(tmp_path):
    bad = 'Design: broken\n```python\ndef score(x):\n    return missing\n```'
    m = method(tmp_path, [response(100), bad, bad, 'no source', answer(1)], budget=5)
    m.run()
    assert [a['status'] for a in m.attempts_table.values()] == ['valid', 'runtime_error', 'known_failure', 'delivery_failed', 'valid']
    assert all(m.attempts_table[i]['repair_of'] == 2 for i in (3, 4, 5))
    assert m.policy['blocks'][0]['spent'] == 4


def test_every_source_gets_same_weak_cross_block_development(tmp_path):
    m = method(tmp_path, [response(1000)] + [answer(i) for i in range(1, 25)], budget=25, frontier_groups=1)
    m.run()
    assert [u['source'] for u in m.policy['units']] == ['Refine', 'Crossover', 'Explore']
    assert [b['purpose'] for b in m.policy['blocks']] == ['new', 'continuation'] * 3
    for u in m.policy['units']:
        assert len(u['trial_ids']) == 8 and len(u['block_ids']) == 2
        fourth, fifth = u['trial_ids'][3:5]
        assert m.attempts_table[fifth]['parent_id'] == fourth
        assert u['closure_reason'] == 'unit_limit'
    assert 'reserved' not in m.policy and 'quota' not in m.policy


def test_late_feedback_and_delivery_error_do_not_change_continuation_rights(tmp_path):
    m = method(tmp_path, [response(1000), 'no source', answer(2, plan=True)] +
               [answer(i) for i in range(3, 9)], budget=9, frontier_groups=1)
    m.run()
    u = m.policy['units'][0]
    assert len(u['trial_ids']) == 8 and u['proposal_id'] == 3
    assert u['question'] == 'change final choice'
    assert all('plan_admission' not in a for a in m.attempts_table.values())
    assert '[Committed Dependency Plan]' not in m.llm.calls[-1][0]


@pytest.mark.parametrize('cut', [1, 4, 5, 6, 9, 10])
def test_resume_at_block_boundaries_matches_uninterrupted_run(tmp_path, cut):
    answers = [response(100)] + [answer(i) for i in range(1, 17)]
    full = method(tmp_path / 'full', answers, budget=17)
    full.run()
    interrupted = method(tmp_path / 'resumed', answers[:cut], budget=17)
    until(interrupted, cut)
    resumed = method(tmp_path / 'resumed', answers[cut:], budget=17)
    resumed.run()
    assert resumed.policy == full.policy
    assert [a['parent_id'] for a in resumed.attempts_table.values()] == [a['parent_id'] for a in full.attempts_table.values()]
    assert [a['prompt_hash'] for a in resumed.attempts_table.values() if 'prompt_hash' in a] == [a['prompt_hash'] for a in full.attempts_table.values() if 'prompt_hash' in a]


def test_abandonment_only_spends_actual_attempts_and_tail_is_one_partial_segment(tmp_path):
    m = method(tmp_path, [response(100), answer(1, status='change_request: abandon')] +
               [answer(i) for i in range(2, 9)], budget=9)
    m.run()
    assert [b['spent'] for b in m.policy['blocks']] == [1, 4, 3]
    assert m.policy['units'][0]['closure_reason'] == 'change_request'
    assert sum(b['spent'] for b in m.policy['blocks']) + m.progress.init_attempts == 9
    assert m.attempts_table[7]['parent_id'] == 6


def test_equal_objective_groups_share_rank_and_boundary_includes_all_ties(tmp_path):
    programs = {i: {'id': i, 'key': str(i), 'fitness': 10.} for i in range(1, 11)}
    vectors = {i: [float(i), float(20-i)] for i in programs}
    vectors[10] = vectors[1]
    records = [{'key': str(i), 'role': 'search', 'valid': True, 'protocol': 'same', 'seed': 1,
                'instances': [{'instance_index': j, 'score': s} for j, s in enumerate(v)]} for i, v in vectors.items()]
    groups = frontier(programs, records, programs, [], 8, 1)
    assert len(groups) == 9 and {g['rank'] for g in groups} == {1}
    assert any(g['members'] == [1, 10] for g in groups)
    assert groups == frontier(programs, list(reversed(records)), reversed(list(programs)), [], 8, 1)
    m = method(tmp_path, [], budget=1)
    for _ in range(20):
        _, selection = m._parent(groups)
        assert selection['group_probability'] == pytest.approx(1 / 9)


def test_event_deltas_reconstruct_checkpoint_and_gains_telescope(tmp_path):
    m = method(tmp_path, [response(1)] + [answer(i) for i in range(2, 22)], budget=21)
    m.run()
    units, blocks, scheduler = {}, {}, {}
    for row in committed_rows(tmp_path):
        delta = row['development']
        scheduler = delta['scheduler']
        if delta['unit']:
            units[delta['unit']['id']] = delta['unit']
            blocks[delta['block']['id']] = delta['block']
    rebuilt = {**scheduler, 'units': list(units.values()), 'blocks': list(blocks.values())}
    assert rebuilt == m.policy == Facts(tmp_path).state['v1024']
    assert sum(b['frontier_gain'] for b in m.policy['blocks']) == 20


def test_control_parser_has_no_resource_license_and_ignores_code():
    fields = declarations('Analysis:\nBase: 999\nPlan: two blocks\nStage 1: x\nDesign: a\nStatus: change_request\n```python\nBase: 1\n```')
    assert fields['base'] == '999' and 'status' not in fields and 'plan' not in fields


@pytest.mark.parametrize('task', CO_TASKS)
def test_taskcard_and_contract_are_available_without_time_targets(task):
    p = PromptBuilder(TokenLLM(), task, training_task(task, condition='traceaad')[0], {}, {}, Config())
    text = p.initial([])['prompt']
    assert '[TaskCard]' in text and 'Lower is better' in text
    assert 'time limit' not in text.lower() and 'seconds' not in text.lower()


def test_resume_rejects_changed_budget_and_old_development_protocol(tmp_path):
    m = method(tmp_path, [response(1)], budget=5)
    until(m, 1)
    with pytest.raises(ValueError, match='same configuration'):
        method(tmp_path, [], budget=6)
    checkpoint = read_json(tmp_path / 'resume.json')
    checkpoint['state']['identity'].pop('development_protocol')
    write_json(tmp_path / 'resume.json', checkpoint)
    with pytest.raises(ValueError, match='same configuration'):
        method(tmp_path, [], budget=5)


def test_experiment_entrypoints(capsys):
    from experiments.traceaad_v10_24 import run
    from experiments.traceaad_v10_24.launch_local import plan_for
    run.main(['--task', 'tsp_construct', '--dry-run'])
    rendered = json.loads(capsys.readouterr().out)
    assert rendered['method'] == 'v1024' and rendered['config']['block_size'] == 4
    assert len(plan_for('test_v1024')) == 18


def test_second_block_local_overflow_does_not_exclude_usable_anchor(tmp_path):
    m = method(tmp_path, [response(1000)] + [answer(i) for i in range(1, 9)], budget=9, frontier_groups=1)
    until(m, 5)
    m.unit['question'] = 'oversized local question ' * 30000
    m._search()  # close first block
    m._search()  # allocate continuation
    m._search()  # local context failure
    assert m.policy['excluded'] == []
    assert m.policy['units'][0]['closure_reason'] == 'local_context_too_long'
    m.run()
    assert m.policy['units'][1]['anchor_id'] == 1 and m.attempts == 9


def test_unusable_base_is_excluded_only_after_minimum_prompt_fails(tmp_path):
    source = 'def score(x):\n    values = [' + ', '.join(str(i) for i in range(4000)) + ']\n    return 1\n'
    m = method(tmp_path, ['Design: large\n```python\n' + source + '```'], budget=5, max_input_tokens=3000)
    result = m.run()
    assert result['budget_used'] == 1 and m.policy['excluded'] == [1]
    assert m.policy['units'][0]['closure_reason'] == 'base_context_too_long'


def test_large_history_is_compacted_and_second_segment_still_runs(tmp_path):
    def large(value):
        lines = '\n'.join(f'    v{i} = {value}' for i in range(180))
        return f'Design: large computation\n```python\ndef score(x):\n{lines}\n    return {value}\n```'
    m = method(tmp_path, [large(i) for i in range(1, 10)], budget=9, max_input_tokens=4000)
    m.run()
    assert m.attempts == 9 and len(m.policy['units'][0]['block_ids']) == 2
    assert not m.policy['excluded']
    assert any(a.get('omitted_materials') for a in m.attempts_table.values())
    assert all(a['input_tokens'] <= 4000 for a in m.attempts_table.values())
    for a, (prompt, _) in zip(list(m.attempts_table.values())[1:], m.llm.calls[1:]):
        assert a['available_bases'] == [a['system_default_base_id']]
        assert m.programs[a['system_default_base_id']]['code'].rstrip() in prompt


def test_producing_request_precedes_later_anchor_reuse(tmp_path):
    m = method(tmp_path, [response(1)] + [answer(i) for i in range(2, 18)], budget=25, frontier_groups=1)
    until(m, 17)
    m._search()  # close second unit
    m._search()  # allocate third; producer of best is unit 2
    producer = m.policy['units'][1]
    assert m.unit['linked_unit_ids'][0] == producer['id']
    request = m.prompts.request(m.unit, m.block, 'Explore', m.unit['worktip_id'], [producer])
    assert request['related_trial_ids'][0] == m.programs[m.unit['anchor_id']]['attempt_id']


def test_diagnostics_report_resources_without_legacy_slots(tmp_path):
    from experiments.infra.diagnose_search import diagnose
    m = method(tmp_path, [response(1)] + [answer(i) for i in range(2, 18)], budget=17)
    m.run()
    d = diagnose(tmp_path)['explorations']
    assert d['kind'] == 'finite_development_requests'
    assert d['candidate_cost_by_purpose'] == {'new': 8, 'continuation': 8}
    assert d['gain_accounting_error'] == 0 and d['opened_requests'] == 2
    assert sum(d['candidate_cost_by_source'].values()) == 16


def test_blank_control_value_does_not_consume_next_field():
    fields = declarations(answer(1))
    assert fields['base'] == '' and fields['change'] == 'score return'


def test_zero_cost_local_failures_end_without_global_program_exclusion(tmp_path):
    from traceaad.common.prompts import ContextTooLong
    m = method(tmp_path, [response(1)], budget=5)
    until(m, 1)
    def unavailable(*args, **kwargs):
        raise ContextTooLong('local materials do not fit')
    m.prompts.request = unavailable
    result = m.run()
    assert result['budget_used'] == 1 and m.policy['excluded'] == []
    assert len(m.policy['units']) == 3
    assert all(b['spent'] == 0 for b in m.policy['blocks'])


def test_equal_donor_complementarity_does_not_create_hash_privilege(tmp_path):
    m = method(tmp_path, [response(1000)] + [answer(i) for i in range(1, 9)], budget=9)
    until(m, 9)
    _, selection = m._donor(1)
    assert len(selection['pool']) == 8 and selection['pool_probability'] == 1
    assert selection['member_probability'] == 1 / 8


def test_oversized_worktip_excludes_itself_and_donor_falls_back_without_a_call(tmp_path):
    source = 'def score(x):\n    values = [' + ', '.join(str(i) for i in range(4000)) + ']\n    return 2\n'
    m = method(tmp_path, [response(1), 'Design: large\n```python\n' + source + '```'] +
               [answer(i) for i in range(3, 10)], budget=9, max_input_tokens=3000)
    until(m, 2)
    m._search()
    assert m.policy['excluded'] == [2] and m.policy['units'][0]['anchor_id'] == 1
    m.run()
    third = m.attempts_table[3]
    assert third['action'] == 'Refine' and third['reference_id'] is None
    assert 'donor_unavailable:refine_fallback' in third['trims']
    assert third['material_ids'] == [1]
    assert m.attempts == m.progress.model_calls == 9
