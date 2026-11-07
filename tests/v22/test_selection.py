"""Check mechanism 1 against independent probability calculations and replay."""

from dataclasses import asdict
import json
import random

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import t

from traceaad_base import Evaluation, Program
from traceaad_v22.selection import Action, OPERATORS, ResponseModel, Selector, expected_improvement


def program(pid, fitness, operator="Init", parent=0, reference=None, code=None):
    return Program(pid, code or f"def solve(x):\n    return x + {pid}\n", "d", operator,
                   parent, reference, Evaluation("success", fitness), analysis="a")


def roots(fitness=(100., 110.)):
    selector = Selector()
    for pid, value in enumerate(fitness, 1):
        selector.observe(pid, None, program(pid, value))
    selector.initialize()
    return selector


def record(selector, action):
    return {**asdict(action), "x": selector.features(action, selector.nodes).tolist()
            if selector.s0 is not None else None,
            "parent_fitness": selector.nodes[action.parent_id].fitness}


def snapshot(selector):
    def default(obj):
        return obj.tolist() if isinstance(obj, np.ndarray) else asdict(obj)
    return json.dumps({"nodes": selector.nodes, "models": {k: {
        **vars(v), "groups": [(asdict(a), asdict(g)) for a, g in v.groups.items()]}
        for k, v in selector.models.items()}, "f0": selector.f0, "s0": selector.s0,
        "champion": selector.champion, "processed": sorted(selector.processed)}, default=default, sort_keys=True)


def test_random_intercept_recursion_matches_augmented_batch_posterior():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(24, 9))
    groups = rng.integers(3, size=len(x))
    y = rng.normal(size=len(x))
    model = ResponseModel()
    actions = [Action(g, "Refine") for g in range(3)]
    for row, group, value in zip(x, groups, y):
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
    assert len(actions) <= 28
