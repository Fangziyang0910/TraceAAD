import random

import pytest

from traceaad_v19_2 import Config, Evaluation, Program
from traceaad_v19_2.models import Attempt
from traceaad_v19_2.selection import (
    active_episode, choose, joint_weights, rank_weights, reference_pool,
    segment_state, source_key,
)


def program(pid, score, parent=0, *, operator='Refine', reference=None, code=None):
    evaluation = Evaluation('success', score) if score is not None else Evaluation('timeout', failure='timeout')
    return Program(pid, code or f'def solve(x):\n    return {pid}\n', f'design{pid}',
                   'Init' if not parent else operator, parent, reference, evaluation,
                   analysis=f'proposal{pid}')


def attempt(aid, parent, child=None, *, operator='Refine', calls=1, episode=1, before=0):
    return Attempt(aid, parent, operator, selection={
        'episode_id': episode, 'episode_anchor_id': parent, 'episode_calls_before': before},
        program_id=child, calls=[{} for _ in range(calls)], completed=True)


def marginal(probabilities):
    return {pid: sum(v for (p, _), v in probabilities.items() if p == pid)
            for pid, _ in probabilities}


def test_source_identity_ignores_only_syntax_trivia():
    assert source_key('def f(x): return x+1') == source_key('# comment\ndef f(x):\n return x + 1\n')
    for different in ('def f(x): return x+2', 'def f(y): return y+1', 'def f(x):\n "doc"\n return x+1'):
        assert source_key(different) != source_key('def f(x): return x+1')


@pytest.mark.parametrize('origin', ['Init', 'Explore', 'Crossover'])
def test_duplicate_birth_does_not_buy_a_segment_or_selection_mass(origin):
    archive = {1: program(1, 10), 2: program(2, 20)}
    config = Config(protect_new_segments=False)
    before, _ = joint_weights(archive, [], config)
    for pid in range(3, 103):
        archive[pid] = program(pid, 10, 0 if origin == 'Init' else 1,
                               operator=origin, code=archive[1].code)
    after, _ = joint_weights(archive, [], config)
    assert after == before
    state = segment_state(archive)
    assert state['frontiers'] == {1: 1, 2: 2}
    assert state['canonical'][102] == 1


def test_old_ancestors_do_not_dilute_frontier_or_inflate_segment_mass():
    archive = {1: program(1, 10), 2: program(2, 20)}
    config = Config(protect_new_segments=False)
    before = marginal(joint_weights(archive, [], config)[0])
    for pid in range(3, 103):
        archive[pid] = program(pid, 10, 1)
    assert marginal(joint_weights(archive, [], config)[0]) == before
    archive[103] = program(103, 9, 102)
    after = marginal(joint_weights(archive, [], config)[0])
    assert set(after) == {2, 103}
    assert after[103] == pytest.approx(before[1])


def test_rank_pressure_survives_archive_growth_and_equal_or_negative_scores():
    archive = {i: program(i, -100 if i == 1 else i) for i in range(1, 1001)}
    weights = marginal(joint_weights(archive, [], Config(protect_new_segments=False))[0])
    assert weights[1] > .57
    original = rank_weights(archive, list(archive))
    for p in archive.values():
        p.evaluation = Evaluation('success', p.evaluation.fitness * 1e8 + 7e11)
    assert rank_weights(archive, list(archive)) == original
    for p in archive.values():
        p.evaluation = Evaluation('success', -12)
    assert list(rank_weights(archive, list(archive)).values()) == pytest.approx([.001] * 1000)


def test_protected_initialization_is_once_per_unique_root_even_if_refine_fails():
    archive = {1: program(1, 1), 2: program(2, 100), 3: program(3, 1, code=program(1, 1).code)}
    probs, _ = joint_weights(archive, [], Config())
    assert probs == {(1, 'Refine'): .5, (2, 'Refine'): .5}
    a = attempt(1, 1, calls=4)
    probs, rows = joint_weights(archive, [a], Config())
    assert probs == {(2, 'Refine'): 1}
    assert rows[(2, 'Refine')]['phase'] == 'initial_development'


def test_working_tip_can_regress_then_recover_and_boundary_keeps_best():
    archive = {1: program(1, 1), 2: program(2, 4, 1), 3: program(3, 6, 2)}
    trials = [attempt(1, 1, 2), attempt(2, 2, 3, before=1)]
    config = Config(protect_new_segments=False, development_calls=3)
    probs, rows = joint_weights(archive, trials, config)
    assert probs == {(3, 'Refine'): 1}
    assert rows[(3, 'Refine')]['episode_anchor_id'] == 1
    assert rows[(3, 'Refine')]['frontier_id'] == 1
    trials.append(attempt(3, 3, calls=1, before=2))
    assert active_episode(archive, trials, config) is None
    assert set(marginal(joint_weights(archive, trials, config)[0])) == {1}


def test_exact_return_to_anchor_updates_tip_without_resetting_allowance():
    archive = {1: program(1, 1), 2: program(2, 4, 1),
               3: program(3, 1, 2, code=program(1, 1).code)}
    state = active_episode(archive, [attempt(1, 1, 2), attempt(2, 2, 3)], Config())
    assert state['parent_id'] == state['episode_anchor_id'] == 1
    assert state['episode_calls_before'] == 2


@pytest.mark.parametrize('operator', ['Explore', 'Crossover'])
def test_new_discovery_is_its_own_anchor_and_receives_only_remaining_calls(operator):
    archive = {1: program(1, 1), 2: program(2, 9, 1, operator=operator)}
    first = attempt(1, 1, 2, operator=operator, calls=2)
    probs, rows = joint_weights(archive, [first], Config(protect_new_segments=False))
    assert probs == {(2, 'Refine'): 1}
    assert rows[(2, 'Refine')]['episode_anchor_id'] == 2
    assert rows[(2, 'Refine')]['episode_calls_before'] == 2
    assert active_episode(archive, [first], Config(development_calls=2)) is None
    probs, _ = joint_weights(archive, [first, attempt(2, 1, calls=2, episode=2)], Config(development_calls=2))
    # The exhausted discovery is not granted another initial-protection block.
    assert any(op != 'Refine' for _, op in probs)


@pytest.mark.parametrize('kind', ['missing', 'failed', 'duplicate'])
def test_failed_or_copied_discovery_does_not_commit_a_block(kind):
    archive = {1: program(1, 1)}
    if kind != 'missing':
        archive[2] = program(2, None if kind == 'failed' else 1, 1, operator='Explore',
                             code=archive[1].code if kind == 'duplicate' else None)
    first = attempt(1, 1, None if kind == 'missing' else 2, operator='Explore')
    assert active_episode(archive, [first], Config()) is None


def test_reference_pool_uses_frontiers_then_distinct_single_segment_fallback():
    archive = {1: program(1, 5), 2: program(2, 4, 1)}
    assert [p.id for p in reference_pool(archive, 2)] == [1]
    archive[3] = program(3, 6)
    archive[4] = program(4, 2, 3)
    assert [p.id for p in reference_pool(archive, 2)] == [4]
    assert reference_pool({1: archive[1]}, 1) == []


def test_logged_joint_conditionals_and_reference_probability_match_policy():
    archive = {i: program(i, i) for i in range(1, 5)}
    config = Config(protect_new_segments=False)
    probabilities, rows = joint_weights(archive, [], config)
    assert sum(probabilities.values()) == pytest.approx(1)
    for pair, p in probabilities.items():
        row = rows[pair]
        assert p == pytest.approx(row['parent_probability'] * row['operator_conditional_probability'])
    for seed in range(20):
        parent, operator, reference, row = choose(archive, [], config, random.Random(seed))
        assert row['joint_probability'] == probabilities[parent.id, operator]
        if reference:
            pool = reference_pool(archive, parent.id)
            assert reference in pool
            assert row['reference_probability'] == 1/len(pool)
