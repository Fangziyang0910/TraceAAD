"""Budget, evidence and recovery contracts of the V10.24 state machine."""

import json

import pytest

from benchmarks.tasks import CO_TASKS, training_task
from tests.support import TinyEvaluation, TokenLLM, response
from traceaad.common.state import Facts
from traceaad.common.storage import committed_rows
from traceaad.v10_24 import Config, TraceAADV1024
from traceaad.v10_24.policy import declarations, frontier, two_block_plan
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


def test_worktip_can_regress_without_overwriting_champion_and_freeze(tmp_path):
    m = method(tmp_path, [response(100), answer(90), answer(80), answer(110), answer(105)], budget=5)
    summary = m.run()
    u = m.policy['units'][0]
    assert (u['anchor_id'], u['proposal_id'], u['champion_id'], u['worktip_id']) == (1, 2, 4, 5)
    assert m.attempts_table[4]['parent_id'] == 3  # continues the weaker implementation
    assert summary['budget_used'] == 5 and summary['best']['id'] == 4
    assert summary['development']['frontier_gain'] == 10
    assert u['anchor_delta'] == 10
    assert len(m.facts.evaluations) == 10  # five search scores and unchanged top-five freeze


def test_failed_proposal_repair_and_sibling_diffs_remain_in_context(tmp_path):
    bad = 'Design: failed computation\n```python\ndef score(x):\n    return missing_value\n```'
    m = method(tmp_path, [response(100), bad, answer(90), answer(110, base='1'), answer(105)], budget=5)
    m.run()
    u = m.policy['units'][0]
    assert u['proposal_id'] == 2 and not m.programs[2]['valid']
    assert m.attempts_table[3]['action'] == 'Repair' and m.attempts_table[3]['repair_of'] == 2
    assert m.attempts_table[4]['parent_id'] == 1  # declared, fully shown base
    prompt = m.llm.calls[-1][0]
    assert 'missing_value' in prompt and 'repair_of=2' in prompt
    assert all(f'Trial {i}:' in prompt for i in (2, 3, 4))
    assert u['pending_failure'] is None and u['worktip_id'] == 5


def test_delivery_failures_duplicates_and_repairs_all_cost_one(tmp_path):
    bad = 'Design: broken\n```python\ndef score(x):\n    return missing\n```'
    m = method(tmp_path, [response(100), 'no source', response(100), bad, answer(1)], budget=5)
    m.run()
    assert [a['status'] for a in m.attempts_table.values()] == ['valid', 'delivery_failed', 'duplicate', 'runtime_error', 'valid']
    assert m.attempts_table[3]['action'] == 'Refine'  # repeat request, no fictional repair input
    assert m.attempts_table[5]['action'] == 'Repair'
    assert m.policy['blocks'][0]['spent'] == 4
    assert m.policy['units'][0]['champion_id'] == 5  # never the duplicate anchor


def test_plan_reserved_before_scores_and_survives_resume(tmp_path):
    answers = [response(100)] + [answer(i) for i in range(1, 9)] + [answer(9, plan=True)] + [answer(i) for i in range(10, 17)]
    full = method(tmp_path / 'full', answers, budget=17, frontier_groups=1)
    full.run()
    interrupted = method(tmp_path / 'resumed', answers[:10], budget=17, frontier_groups=1)
    original_prepare = interrupted._prepare_attempt
    seen = []
    def prepare(request, details, parent):
        result = original_prepare(request, details, parent)
        if request.get('plan_admission') == 'accepted_before_evaluation':
            seen.append((interrupted.attempts, len(interrupted.facts.evaluations)))
        return result
    interrupted._prepare_attempt = prepare
    until(interrupted, 10)
    assert seen == [(9, 9)]
    assert interrupted.policy['reserved'] == 4
    resumed = method(tmp_path / 'resumed', answers[10:], budget=17, frontier_groups=1)
    resumed.run()
    assert resumed.policy == full.policy
    assert resumed.policy['reserved'] == 0
    u = resumed.policy['units'][2]
    assert len(u['trial_ids']) == 8 and len(u['block_ids']) == 2
    assert resumed.policy['blocks'][3]['allocation_reason'] == 'precommitted'
    assert [a['parent_id'] for a in resumed.attempts_table.values()] == [a['parent_id'] for a in full.attempts_table.values()]


def test_no_plan_when_second_block_cannot_fit_and_no_weak_renewal(tmp_path):
    answers = [response(100)] + [answer(i, plan=i == 9) for i in range(1, 13)]
    m = method(tmp_path / 'short', answers, budget=13, frontier_groups=1)
    m.run()
    assert m.attempts_table[10]['plan_admission'] == 'single_block'
    m = method(tmp_path / 'weak', [response(100)] + [answer(i) for i in range(1, 17)], budget=17, frontier_groups=1)
    m.run()
    assert m.policy['blocks'][-1]['allocation_reason'] == 'no_eligible_continuation'
    assert m.policy['blocks'][-1]['purpose'] == 'front_refine'
    assert len(m.policy['units'][2]['trial_ids']) == 4


def test_change_request_releases_reservation_and_unused_quota_exactly_once(tmp_path):
    answers = [response(100)] + [answer(i) for i in range(1, 9)]
    answers += [answer(9, plan=True, status='change_request: the question is inapplicable')]
    answers += [answer(i) for i in range(10, 17)]
    m = method(tmp_path, answers, budget=17, frontier_groups=1)
    m.run()
    u = m.policy['units'][2]
    assert u['status'] == 'closed' and u['closure_reason'] == 'change_request'
    assert len(u['trial_ids']) == 1 and m.policy['reserved'] == 0
    assert m.policy['blocks'][3]['allocation_reason'] == 'unused_allocation'
    assert m.policy['blocks'][3]['limit'] == 3
    assert sum(b['spent'] for b in m.policy['blocks']) + m.progress.init_attempts == 17


def test_groups_use_full_vectors_and_do_not_multiply_equal_observations():
    programs = {i: {'id': i, 'key': str(i), 'fitness': 2.} for i in range(1, 5)}
    vectors = {1: [1., 3.], 2: [1., 3.], 3: [3., 1.], 4: [2., 2.]}
    records = [{'key': str(i), 'role': 'search', 'valid': True, 'protocol': 'same', 'seed': 1,
                'instances': [{'instance_index': j, 'score': s} for j, s in enumerate(v)]} for i, v in vectors.items()]
    a = frontier(programs, records, programs, [], 8, 1)
    b = frontier(programs, list(reversed(records)), reversed(list(programs)), [], 8, 1)
    assert a == b and len(a) == 3
    assert any(g['members'] == [1, 2] for g in a)


def test_event_deltas_reconstruct_checkpoint_and_budget_gains_telescope(tmp_path):
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
    assert all(len(u['block_ids']) <= 2 for u in m.policy['units'])


def test_control_parser_ignores_code_and_non_analysis_text():
    fields = declarations('Analysis:\nBase: 999\nPlan: two blocks\nStage 1: x\nDesign: a\nStatus: change_request\n```python\nBase: 1\n```')
    assert fields['base'] == '999' and 'status' not in fields
    assert not two_block_plan(fields)


@pytest.mark.parametrize('task', CO_TASKS)
def test_taskcard_and_contract_are_available_without_time_targets(task):
    p = PromptBuilder(TokenLLM(), task, training_task(task, condition='traceaad')[0], {}, {}, Config())
    text = p.initial([])['prompt']
    assert '[TaskCard]' in text and 'Lower is better' in text
    assert 'time limit' not in text.lower() and 'seconds' not in text.lower()


def test_resume_rejects_changed_budget(tmp_path):
    m = method(tmp_path, [response(1)], budget=5)
    until(m, 1)
    with pytest.raises(ValueError, match='same configuration'):
        method(tmp_path, [], budget=6)


def test_experiment_entrypoints(capsys):
    from experiments.traceaad_v10_24 import run
    from experiments.traceaad_v10_24.launch_local import plan_for
    run.main(['--task', 'tsp_construct', '--dry-run'])
    rendered = json.loads(capsys.readouterr().out)
    assert rendered['method'] == 'v1024' and rendered['config']['block_size'] == 4
    plan = plan_for('test_v1024')
    assert len(plan) == 18
    assert all('experiments.traceaad_v10_24.run' in row['command'] for row in plan)


def test_context_overflow_preserves_required_material_and_ends_without_free_candidates(tmp_path):
    from traceaad.common.prompts import ContextTooLong
    m = method(tmp_path, [response(1)], budget=5)
    until(m, 1)
    def unavailable(*args, **kwargs):
        raise ContextTooLong('required evidence cannot fit')
    m.prompts.request = unavailable
    summary = m.run()
    assert summary['status'] == 'finished' and summary['budget_used'] == 1
    assert m.policy['blocks'][0]['completion_reason'] == 'required_context_too_long'
    assert m.policy['blocks'][0]['spent'] == 0


def test_frontier_eligibility_is_rechecked_and_member_need_not_be_selected(tmp_path):
    m = method(tmp_path, [response(1)] + [answer(i) for i in range(2, 18)], budget=17, frontier_groups=1)
    m.run()
    u = m.policy['units'][2]
    assert len(u['block_ids']) == 2
    assert m.policy['blocks'][3]['allocation_reason'] == 'current_frontier'
    assert u['events']['first_frontier'] == 10


def test_diagnostics_include_real_costs_and_not_legacy_exploration_counts(tmp_path):
    from experiments.infra.diagnose_search import diagnose
    m = method(tmp_path, [response(1)] + [answer(i) for i in range(2, 18)], budget=17)
    m.run()
    result = diagnose(tmp_path)
    assert result['actions']['Develop']['attempts'] > 0
    d = result['explorations']
    assert d['kind'] == 'finite_development_requests'
    assert sum(d['candidate_cost_by_purpose'].values()) == 16
    assert d['gain_accounting_error'] == 0


def test_short_tail_uses_single_attempt_refinements(tmp_path):
    m = method(tmp_path, [response(1)] + [answer(i) for i in range(2, 9)], budget=8)
    m.run()
    assert [b['spent'] for b in m.policy['blocks']] == [4, 1, 1, 1]
    assert all(m.attempts_table[i]['action'] == 'Refine' for i in (6, 7, 8))


def test_source_less_plan_is_not_a_commitment(tmp_path):
    no_code = ('Analysis:\nPlan: two blocks\nStage 1: build a table\nStage 2: use the table\n'
               'Dependency: table required\nLocation: score\n')
    m = method(tmp_path, [response(100)] + [answer(i) for i in range(1, 9)] + [no_code], budget=17)
    until(m, 10)
    assert m.policy['reserved'] == 0 and m.attempts_table[10]['status'] == 'delivery_failed'


def test_promised_stages_are_in_every_followup_context(tmp_path):
    answers = [response(100)] + [answer(i) for i in range(1, 9)] + [answer(9, plan=True)] + [answer(i) for i in range(10, 17)]
    m = method(tmp_path, answers, budget=17, frontier_groups=1)
    m.run()
    for prompt, _ in m.llm.calls[10:17]:
        assert '[Committed Dependency Plan]' in prompt
        assert 'connect estimates' in prompt and 'output needs the new estimates' in prompt


def test_reserved_tail_can_cross_nominal_slots_without_negative_quota(tmp_path):
    answers = [response(100)] + [answer(i, plan=i == 29) for i in range(1, 40)]
    m = method(tmp_path, answers, budget=40, frontier_groups=1)
    m.run()
    promised = next(u for u in m.policy['units'] if u.get('plan'))
    assert len(promised['trial_ids']) == 8
    assert m.policy['reserved'] == 0 and m.policy['quota'] == 1
    tail = m.policy['blocks'][-1]
    assert tail['allocation_reason'] == 'reserved_tail' and tail['spent'] == 4
    assert [a['spent'] for a in tail['nominal_allocations']] == [1, 3]
    assert m.policy['slot'] * 4 + 4 - m.policy['quota'] == 39
