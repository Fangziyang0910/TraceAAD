"""Check mechanism 1 against independent probability calculations and replay."""

from dataclasses import asdict
import json
import random

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import stdtrit
from scipy.stats import t

from traceaad_base import Evaluation, Program
from traceaad_v22.selection import (Action, OPERATORS, QUANTILES, ResponseModel, Selector,
                                    expected_improvement)


def program(pid, fitness, operator="Init", parent=0, reference=None, code=None):
    return Program(pid, code or f"def solve(x):\n    return x + {pid}\n", "d", operator,
                   parent, reference, Evaluation("success", fitness), analysis="a")


def roots(fitness=(100., 110.), residual_clip=3.):
    selector = Selector(residual_clip)
    for pid, value in enumerate(fitness, 1):
        selector.observe(pid, None, program(pid, value))
    selector.initialize()
    return selector


def record(selector, action):
    return {**asdict(action), "x": selector.features(action, selector.nodes).tolist()
            if selector.s0 is not None else None,
            "parent_fitness": selector.nodes[action.parent_id].fitness,
            "f0": selector.f0, "s0": selector.s0}


def snapshot(selector):
    def default(obj):
        return obj.tolist() if isinstance(obj, np.ndarray) else asdict(obj)
    return json.dumps({"nodes": selector.nodes, "models": {k: {
        **vars(v), "groups": [(asdict(a), asdict(g)) for a, g in v.groups.items()]}
        for k, v in selector.models.items()}, "f0": selector.f0, "s0": selector.s0,
        "champion": selector.champion, "processed": sorted(selector.processed)}, default=default, sort_keys=True)


def test_unclipped_random_intercept_recursion_matches_augmented_batch_posterior():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(24, 9))
    groups = rng.integers(3, size=len(x))
    y = .2 * rng.normal(size=len(x))
    model = ResponseModel()
    actions = [Action(g, "Refine") for g in range(3)]
    for row, group, value in zip(x, groups, y):
        _, mu, scale, _ = model.predict([actions[group]], row[None])
        assert abs(value - mu[0]) < 3 * scale[0]
        model.observe_gate(actions[group], True)
        model.observe_usable(actions[group], row, value)
    design = np.c_[x, np.eye(3)[groups]]
    precision = np.diag([1.] + [4.] * 8 + [1.] * 3)
    covariance = np.linalg.inv(precision + design.T @ design)
    mean = covariance @ design.T @ y
    B = 2. + .5 * (y @ y - mean @ (precision + design.T @ design) @ mean)
    np.testing.assert_allclose(model.m, mean[:9], atol=1e-14)
    np.testing.assert_allclose(model.V, covariance[:9, :9], atol=1e-14)
    assert model.B == pytest.approx(B)
    for group in range(4):
        row = rng.normal(size=9)
        _, mu, scale, df = model.predict([Action(group, "Refine")], row[None])
        augmented = np.r_[row, np.eye(3)[group] if group < 3 else np.zeros(3)]
        variance = 1. + augmented @ covariance @ augmented + (group == 3)
        assert mu[0] == pytest.approx(augmented @ mean)
        assert scale[0] ** 2 == pytest.approx(B / model.A * variance)
        assert df == 6 + len(y)


@pytest.mark.parametrize("mu,scale,df,gap", [(-.4, 1.3, 6., .7), (3., .2, 8., 1.), (-2., 2., 20., 5.)])
def test_student_t_improvement_matches_direct_integration(mu, scale, df, gap):
    direct = quad(lambda y: (y - gap) * t.pdf((y - mu) / scale, df) / scale, gap, np.inf)[0]
    assert expected_improvement(.7, mu, scale, df, gap) == pytest.approx(.7 * direct)


def test_gate_excludes_own_group_and_reference_changes_group():
    model = ResponseModel()
    a, b, changed = Action(1, "Crossover", 2), Action(2, "Crossover", 1), Action(1, "Crossover", 3)
    for usable in [True, True, False]:
        model.observe_gate(a, usable)
    model.observe_gate(b, False)
    pi, *_ = model.predict([a, changed], np.ones((2, 9)))
    assert pi[0] == pytest.approx((4 * (1 / 3) + 2) / 7)
    assert pi[1] == pytest.approx(3 / 6)
    assert changed not in model.groups


def test_pool_identity_families_and_signed_parent_improvement():
    selector = roots()
    action = Action(1, "Explore")
    selector.observe(3, record(selector, action), program(3, 115., "Explore", 1))
    action = Action(3, "Refine")
    selector.observe(4, record(selector, action), program(4, 105., "Refine", 3))
    assert selector.nodes[4].family == selector.nodes[3].family == 3
    assert selector.nodes[4].birth_change == 10.
    assert selector.nodes[4].ancestor_best == 100.
    assert selector.models['Refine'].groups[action].sy == 1.
    action = Action(4, "Crossover", 2)
    selector.observe(5, record(selector, action), program(5, 108., "Crossover", 4, 2))
    assert selector.nodes[5].family == 5
    assert selector.nodes[5].ancestor_best == 100.
    duplicate = program(6, 105., "Refine", 4, code=program(4, 105.).code + "\n# same AST\n")
    action = Action(4, "Refine")
    r = record(selector, action)
    selector.observe(6, r, duplicate)
    assert not r['usable'] and 6 not in selector.nodes
    assert selector.models['Refine'].groups[action].k == 0
    assert selector.models['Refine'].groups[action].n == 1
    selector.observe(7, None, program(7, float('inf')))
    assert 7 not in selector.nodes


def test_scale_preserves_pair_multiplicity():
    selector = roots((0., 0., 1., 2.))
    assert selector.f0 == 0. and selector.s0 == 1.


def test_tied_initialization_freezes_later_and_replays_regression_once():
    selector = roots((100., 100.))
    action, r = selector.choose(10, random.Random(2))
    assert selector.s0 is None and r['x'] is None and r['I'] is None
    selector.observe(3, r, program(3, 105., action.operator, action.parent_id, action.reference_id))
    assert selector.f0 == 100. and selector.s0 == 5.
    model = selector.models[action.operator]
    assert model.N == model.U == 1 and model.groups[action].k == 1
    assert model.groups[action].sy == -1.
    assert selector.nodes[3].momentum == -.25
    before = snapshot(selector)
    selector.observe(3, r, program(3, 105., action.operator, action.parent_id, action.reference_id))
    assert snapshot(selector) == before


def test_virtual_child_becomes_reference_with_a_fresh_group():
    selector = roots()
    old = Action(1, 'Crossover', 2)
    selector.models['Crossover'].observe_gate(old, False)
    child = selector._node(3, 95., 'Explore', 1, None)
    actions, nodes = selector.second_actions([old], Action(1, 'Explore'), selector.family_champions(), child)
    new = Action(1, 'Crossover', 3)
    assert new in actions and old not in actions
    assert selector.features(new, nodes)[8] == pytest.approx(1 / 3)
    assert new not in selector.models['Crossover'].groups
    assert Action(3, 'Crossover', 1) in actions
    # A refined child inherits its parent's family and cannot be its reference.
    child = selector._node(3, 95., 'Refine', 1, None)
    actions, _ = selector.second_actions([old], Action(1, 'Refine'), selector.family_champions(), child)
    assert old in actions and Action(1, 'Crossover', 3) not in actions
    assert Action(3, 'Crossover', 2) in actions


def test_shadow_updates_do_not_touch_real_state():
    selector = roots()
    before = snapshot(selector)
    action, r = selector.choose(8, random.Random(0))
    assert snapshot(selector) == before
    assert r['Q'] > r['I'] and action.parent_id in selector.nodes


def test_two_step_can_prefer_a_regressing_exploration(monkeypatch):
    selector = roots((100., 110.))
    def forecasts(actions, nodes, models, champion):
        child = next((n for pid, n in nodes.items() if pid > 2), None)
        rows = []
        for action in actions:
            if child is None:
                value = 1. if action.operator == 'Refine' else 0.
            else:
                value = 5. if child.operator == 'Explore' and action.parent_id == child.id else 1.
            rows.append([1., -1., .01, 6., value])
        return np.array(rows)
    monkeypatch.setattr(selector, 'forecasts', forecasts)
    action, r = selector.choose(4, random.Random(0))
    assert action.operator == 'Explore' and r['I'] == 0. and r['Q'] == pytest.approx(5.)


@pytest.mark.parametrize('remaining', [1, 2, 3])
def test_short_budget_uses_only_one_step(monkeypatch, remaining):
    selector = roots()
    def forbidden(*args):
        raise AssertionError('Two-step planning with insufficient budget')
    monkeypatch.setattr(selector, 'two_step', forbidden)
    _, r = selector.choose(remaining, random.Random(0))
    assert r['Q'] == r['I'] and r['remaining_evaluations'] == remaining


def test_second_step_candidate_limit_and_family_tie_break():
    selector = roots(tuple(100. + i // 2 for i in range(30)))
    references = selector.family_champions()
    assert [n.id for n in references] == [1, 2]
    assert selector.actions_for(selector.nodes[1], references)[2].reference_id == 2
    base = [Action(i, operator, 1 if i != 1 else 2) if operator == 'Crossover' else Action(i, operator)
            for operator in OPERATORS for i in range(1, 9)]
    action = Action(30, 'Explore')
    child = selector._node(31, 90., action.operator, action.parent_id, action.reference_id)
    actions, _ = selector.second_actions(base, action, references, child)
    assert len(actions) == 29
    assert Action(30, 'Crossover', 31) in actions


@pytest.mark.parametrize('sign', [-1., 1.])
def test_single_extreme_response_bounds_mean_scale_and_local_sum(sign):
    action = Action(1, 'Explore')
    x = np.array([1.] + [0.] * 8)
    outcomes = []
    for magnitude in [1e3, 1e9]:
        model = ResponseModel()
        _, mu, scale, _ = model.predict([action], x[None])
        effective_y = mu[0] + sign * 3. * scale[0]
        model.observe_gate(action, True)
        model.observe_usable(action, x, sign * magnitude)
        assert model.groups[action].sy == pytest.approx(effective_y)
        assert model.m[0] == pytest.approx(effective_y / 3.)
        assert model.B == pytest.approx(5.)
        assert model.A == 3.5 and model.groups[action].k == 1
        outcomes.append(model)
    np.testing.assert_array_equal(outcomes[0].m, outcomes[1].m)
    np.testing.assert_array_equal(outcomes[0].V, outcomes[1].V)
    assert outcomes[0].B == outcomes[1].B


def test_clip_uses_old_prediction_and_effective_level_not_raw_residual():
    model = ResponseModel()
    action = Action(1, 'Refine')
    x = np.array([1., .3] + [0.] * 7)
    for y in [1., 1.5, 2.]:
        model.observe_gate(action, True)
        model.observe_usable(action, x, y)
    _, mu, scale, _ = model.predict([action], x[None])
    group = model.groups[action]
    c = 1. / (group.k + 1.)
    z = x - c * group.sx
    H = 1. + c + z @ model.V @ z
    e = -3 * scale[0]
    expected_m = model.m + (model.V @ z) * e / H
    expected_B = model.B + e * e / (2 * H)
    expected_sy = group.sy + mu[0] + e
    model.observe_gate(action, True)
    model.observe_usable(action, x, -1e8)
    np.testing.assert_allclose(model.m, expected_m)
    assert model.B == pytest.approx(expected_B)
    assert model.groups[action].sy == pytest.approx(expected_sy)


@pytest.mark.parametrize('fitness', [-1e6, 1e6])
def test_clipped_learning_keeps_true_champion_node_and_trajectory(fitness):
    selector = roots()
    action = Action(1, 'Explore')
    selector.observe(3, record(selector, action), program(3, fitness, 'Explore', 1))
    assert selector.champion == min(100., fitness)
    child = selector.nodes[3]
    assert child.fitness == fitness and child.birth_change == 100. - fitness
    assert child.family == 3 and child.ancestor_best == min(100., fitness)
    assert child.momentum == pytest.approx(.5 * ((100. - fitness) / 10.) / (1 + abs((100. - fitness) / 10.)))
    assert abs(selector.models['Explore'].groups[action].sy) == pytest.approx(3 * np.sqrt(2.))


def test_virtual_children_use_raw_samples_and_same_clipped_update(monkeypatch):
    selector = roots(residual_clip=.01)
    action = Action(1, 'Explore')
    prediction = selector.forecasts([action], selector.nodes, selector.models, selector.champion)[0]
    ys = prediction[1] + prediction[2] * stdtrit(prediction[3], QUANTILES)
    before = snapshot(selector)
    children, updates = [], []
    original_node, original_update = selector._node, ResponseModel.observe_usable
    def node(*args):
        children.append(args[1])
        return original_node(*args)
    def update(model, action, x, y):
        _, mu, scale, _ = model.predict([action], x[None])
        original_update(model, action, x, y)
        assert model.groups[action].sy == pytest.approx(mu[0] + np.clip(y - mu[0], -.01 * scale[0], .01 * scale[0]))
        updates.append(float(y))
    monkeypatch.setattr(selector, '_node', node)
    monkeypatch.setattr(ResponseModel, 'observe_usable', update)
    selector.two_step(action, prediction, [Action(1, 'Refine')], selector.family_champions())
    np.testing.assert_allclose(updates, ys)
    np.testing.assert_allclose(children, 100. - 10. * ys)
    assert snapshot(selector) == before


def test_first_other_family_adds_current_parent_c_without_expanding_other_parents():
    selector = roots((100.,))
    refine = Action(1, 'Refine')
    selector.observe(2, record(selector, refine), program(2, 110., 'Refine', 1))
    action = Action(1, 'Explore')
    child = selector._node(3, 105., 'Explore', 1, None)
    base = [Action(i, op) for i in [1, 2] for op in OPERATORS[:2]]
    actions, _ = selector.second_actions(base, action, selector.family_champions(), child)
    assert Action(1, 'Crossover', 3) in actions
    assert Action(2, 'Crossover', 3) not in actions
    assert Action(3, 'Crossover', 1) in actions
    assert len(actions) == 8


def test_supplemented_pair_respects_existing_best_reference_rule():
    selector = roots()
    action = Action(2, 'Explore')
    child = selector._node(3, 115., 'Explore', 2, None)
    actions, _ = selector.second_actions([Action(1, 'Refine')], action, selector.family_champions(), child)
    assert Action(2, 'Crossover', 3) not in actions
    assert Action(3, 'Crossover', 1) in actions
    refined = selector._node(3, 90., 'Refine', 2, None)
    actions, _ = selector.second_actions([], Action(2, 'Refine'), selector.family_champions(), refined)
    assert Action(2, 'Crossover', 3) not in actions


@pytest.mark.parametrize('threshold', [0., -1., float('inf'), float('nan')])
def test_residual_threshold_must_be_positive_and_finite(threshold):
    with pytest.raises(ValueError, match='residual_clip'):
        Selector(threshold)
