"""Check the Base-anchored champion-yield estimator against hand calculations."""

import random
from itertools import product
from math import prod

import pytest

from traceaad_base import Config, Evaluation, Program
from traceaad_base.selection import random_references, reference_pool, weights
from traceaad_v22_2.selection import MULTIPLIER_CAP, Action, AttemptRecord, Estimate, Selector


def program(pid, fitness, operator="Init", parent=0, reference=None, code=None, status="success"):
    return Program(
        pid,
        code or f"def solve(x):\n    return x + {pid}\n",
        "d",
        operator,
        parent,
        reference,
        Evaluation(
            status,
            fitness if status == "success" else None,
            failure="err" if status != "success" else None,
        ),
        analysis="a",
    )


def roots(fitness=(100.0, 110.0)):
    selector = Selector()
    for pid, value in enumerate(fitness, 1):
        selector.observe(pid, None, program(pid, value), evaluations=1, model_calls=1)
    return selector


def references_for(selector):
    return random_references({pid: node.fitness for pid, node in selector.nodes.items()}, random.Random(0))


def allocation(selector, config=None, *, first_only=False):
    return selector.allocation(config, first_only=first_only, reference_ids=references_for(selector))


def record_for(selector, action):
    parent = selector.nodes[action.parent_id]
    return {
        "parent_id": action.parent_id,
        "operator": action.operator,
        "reference_id": action.reference_id,
        "parent_fitness": parent.fitness,
        "reference_fitness": (
            selector.nodes[action.reference_id].fitness if action.reference_id is not None else None
        ),
        "champion_before_attempt": selector.champion,
        "evaluations_used_before_attempt": selector.evaluations_used,
        "parent_origin": parent.operator,
        "parent_birth_change": parent.birth_change,
        "parent_momentum": parent.momentum,
        "parent_ancestor_best": parent.ancestor_best,
    }


def do(selector, attempt_id, action, child, *, evaluations=1, first_result=None):
    record = record_for(selector, action)
    selector.observe(
        attempt_id, record, child, evaluations=evaluations, model_calls=1, first_result=first_result
    )
    return record


ERROR_FIRST = {
    "status": "error",
    "fitness": None,
    "source_identity": None,
    "usable_at_attempt_start": False,
}


def inject(
    selector,
    attempt_id,
    *,
    action,
    parent_fitness,
    champion_before,
    gain,
    evaluations_before,
    evaluations=1,
    first_gain=None,
):
    selector.history.append(
        AttemptRecord(
            attempt_id=attempt_id,
            action=action,
            parent_fitness=parent_fitness,
            reference_fitness=None,
            champion_before=champion_before,
            evaluations_before=evaluations_before,
            parent_origin="Init",
            parent_birth_change=0.0,
            parent_momentum=0.0,
            parent_ancestor_best=parent_fitness,
            champion_gain=gain,
            first_champion_gain=gain if first_gain is None else first_gain,
            evaluations=evaluations,
            model_calls=1,
            first_result={},
            final_result={},
        )
    )


def test_node_longitudinal_quantities_and_family():
    selector = roots((100.0, 110.0))
    assert selector.nodes[1].birth_change == 0.0
    assert selector.nodes[1].momentum == 0.0
    assert selector.nodes[1].ancestor_best == 100.0
    assert selector.nodes[1].family == 1

    do(selector, 3, Action(1, "Explore"), program(3, 105.0, "Explore", 1))
    node3 = selector.nodes[3]
    assert node3.birth_change == -5.0
    assert node3.momentum == pytest.approx(-0.5)
    assert node3.ancestor_best == 100.0
    assert node3.family == 3

    do(selector, 4, Action(3, "Refine"), program(4, 102.0, "Refine", 3))
    node4 = selector.nodes[4]
    assert node4.birth_change == 3.0
    assert node4.momentum == pytest.approx(0.25)
    assert node4.family == 3

    do(selector, 5, Action(4, "Refine"), program(5, 98.0, "Refine", 4))
    node5 = selector.nodes[5]
    assert node5.momentum == pytest.approx(0.625)
    assert node5.ancestor_best == 98.0

    do(selector, 6, Action(5, "Crossover", 2), program(6, 99.0, "Crossover", 5, 2))
    node6 = selector.nodes[6]
    assert node6.family == 6
    assert node6.ancestor_best == 98.0
    assert node6.birth_change == -1.0
    assert node6.momentum == pytest.approx(-0.5 + 0.5 * 0.625)


def test_champion_gain_is_direct_not_parent_gain():
    selector = roots((100.0, 120.0))
    record = do(selector, 3, Action(2, "Refine"), program(3, 110.0, "Refine", 2))
    assert record["champion_gain"] == 0.0
    assert 3 in selector.nodes

    record = do(selector, 4, Action(1, "Refine"), program(4, 99.0, "Refine", 1))
    assert record["champion_gain"] == 1.0

    do(selector, 5, Action(3, "Refine"), program(5, 105.0, "Refine", 3))
    record = do(selector, 6, Action(5, "Refine"), program(6, 95.0, "Refine", 5))
    assert record["champion_gain"] == 4.0

    duplicate = program(7, 95.0, "Refine", 5, code=program(6, 95.0).code + "\n# same AST\n")
    record = do(selector, 7, Action(5, "Refine"), duplicate)
    assert record["champion_gain"] == 0.0
    assert 7 not in selector.nodes

    record = do(selector, 8, Action(5, "Explore"), program(8, None, "Explore", 5, status="error"))
    assert record["champion_gain"] == 0.0
    assert 8 not in selector.nodes


def test_estimator_matches_hand_calculation():
    selector = roots((100.0, 110.0))
    selector.evaluations_used = 100
    inject(
        selector,
        11,
        action=Action(901, "Refine"),
        parent_fitness=100.0,
        champion_before=100.0,
        gain=4.0,
        evaluations_before=90,
        evaluations=1,
    )
    inject(
        selector,
        12,
        action=Action(902, "Refine"),
        parent_fitness=110.0,
        champion_before=100.0,
        gain=0.0,
        evaluations_before=90,
        evaluations=2,
    )
    inject(
        selector,
        13,
        action=Action(903, "Refine"),
        parent_fitness=100.0,
        champion_before=90.0,
        gain=2.0,
        evaluations_before=80,
        evaluations=1,
    )
    inject(
        selector,
        14,
        action=Action(1, "Refine"),
        parent_fitness=100.0,
        champion_before=100.0,
        gain=1.0,
        evaluations_before=95,
        evaluations=1,
    )

    estimate = selector.estimate_action(Action(1, "Refine"))
    # Pairwise distances among the three shared states are all 10, so h = 20.
    # Quality weights are 1, 0.5, 0.5; trajectory weights are 1; recency uses H = 256.
    w1 = 1.0 * (1 - 10 / 256)
    w2 = 0.5 * (1 - 10 / 256)
    w3 = 0.5 * (1 - 20 / 256)
    w_local = 1 - 5 / 256
    n_shared = w1 + w2 + w3
    y_shared = w1 * 4.0 + w3 * 2.0
    c_shared = w1 + w2 * 2.0 + w3
    expected_value = (w_local * 1.0 + y_shared) / (w_local * 1.0 + c_shared)
    effective = w_local + min(n_shared, 16.0)
    assert estimate.matching_radius == pytest.approx(20.0)
    assert estimate.shared_weight == pytest.approx(n_shared)
    assert estimate.local_weight == pytest.approx(w_local)
    assert estimate.value == pytest.approx(expected_value)
    assert estimate.eta == pytest.approx(effective / (16.0 + effective))
    assert estimate.window_breakthroughs == 3


def test_local_breakthrough_survives_its_own_champion_jump():
    selector = roots((100.0, 110.0))
    do(selector, 3, Action(1, "Refine"), program(3, 80.0, "Refine", 1))
    estimate = selector.estimate_action(Action(1, "Refine"))
    assert estimate.value == pytest.approx(20.0)
    assert estimate.local_weight == pytest.approx(255 / 256)
    assert estimate.eta > 0.0


def test_far_shared_breakthrough_gets_zero_quality_weight():
    selector = roots((100.0, 110.0))
    selector.evaluations_used = 100
    for attempt_id, parent_fitness in ((11, 200.0), (12, 202.0), (13, 204.0)):
        inject(
            selector,
            attempt_id,
            action=Action(900 + attempt_id, "Refine"),
            parent_fitness=parent_fitness,
            champion_before=190.0,
            gain=50.0,
            evaluations_before=90,
        )
    estimate = selector.estimate_action(Action(1, "Refine"))
    assert estimate.value == 0.0
    assert estimate.shared_weight == 0.0
    assert estimate.eta == 0.0


def test_scale_and_shift_invariance_when_temperature_uses_the_same_units():
    def build(scale, shift):
        def value(v):
            return scale * v + shift

        selector = Selector()
        for pid, fitness in [(1, value(100.0)), (2, value(110.0))]:
            selector.observe(pid, None, program(pid, fitness), evaluations=1, model_calls=1)
        do(selector, 3, Action(1, "Refine"), program(3, value(96.0), "Refine", 1))
        do(
            selector,
            4,
            Action(2, "Explore"),
            program(4, value(92.0), "Explore", 2),
            evaluations=2,
            first_result=ERROR_FIRST,
        )
        actions, _, _, probs, _ = allocation(selector, Config(selection_temperature=scale))
        return {(a.parent_id, a.operator, a.reference_id): p for a, p in zip(actions, probs)}

    original = build(1.0, 0.0)
    transformed = build(7.5, -320.0)
    assert original.keys() == transformed.keys()
    for key in original:
        assert original[key] == pytest.approx(transformed[key], rel=1e-12, abs=1e-12)


def test_missing_reference_is_an_error():
    selector = roots((100.0,))
    with pytest.raises(RuntimeError, match="No reference"):
        allocation(selector)
    actions, beta, _, _, _ = allocation(selector, Config(crossover_weight=0.0))
    assert [a.operator for a in actions] == ["Refine", "Explore"]
    assert beta == pytest.approx([0.45 / 0.75, 0.30 / 0.75])


def test_first_view_ignores_repair_breakthrough():
    selector = roots((100.0, 110.0))
    record = do(
        selector,
        3,
        Action(1, "Refine"),
        program(3, 90.0, "Refine", 1),
        evaluations=2,
        first_result=ERROR_FIRST,
    )
    assert record["first_champion_gain"] == 0.0
    assert record["champion_gain"] == 10.0
    full = selector.estimate_action(Action(1, "Refine"))
    first = selector.estimate_action(Action(1, "Refine"), first_only=True)
    assert full.value == pytest.approx(5.0)
    assert first.value == 0.0
    _, chosen = selector.choose(1, random.Random(0), reference_ids=references_for(selector))
    assert chosen["valuation_view"] == "first_result"


def test_multiplier_cap_bounds_single_positive_dominance():
    selector = roots((100.0, 105.0, 110.0, 115.0, 120.0))
    actions = selector.actions(references_for(selector))
    crafted = [Estimate(0.0, 0.5, 10.0, 0.0, 1.0, 0)] * (len(actions) - 1) + [
        Estimate(100.0, 0.5, 10.0, 0.0, 1.0, 1)
    ]
    selector._estimates = lambda given, **kwargs: crafted
    given, beta, _, probs, diagnostics = allocation(selector)
    assert given == actions
    assert diagnostics["uncapped_multipliers"][-1] > MULTIPLIER_CAP
    assert diagnostics["multipliers"][-1] == MULTIPLIER_CAP
    assert sum(probs) == pytest.approx(1.0)
    assert probs[-1] / beta[-1] < diagnostics["uncapped_multipliers"][-1]


def test_zero_evaluation_record_keeps_cost_without_penalizing_base_probability():
    selector = roots((100.0, 110.0, 120.0))
    _, before, _, _, _ = allocation(selector)
    record = record_for(selector, Action(1, "Refine"))
    selector.observe(4, record, None, evaluations=0, model_calls=2)
    assert len(selector.history) == 1
    assert selector.history[0].evaluations == 0
    estimate = selector.estimate_action(Action(1, "Refine"))
    assert estimate.value == 0.0 and estimate.eta == 0.0
    assert selector.history[0].model_calls == 2
    _, after, _, _, _ = allocation(selector)
    assert after == before


def test_mechanism_and_base_share_boltzmann_quality_prior():
    selector = roots((5.69, 5.70, 5.80))
    config = Config(selection_temperature=0.1)
    archive = {pid: program(pid, node.fitness) for pid, node in selector.nodes.items()}
    quality = weights(archive, config.selection_temperature)
    normalizer = sum(quality.values())
    actions, beta, estimates, probabilities, _ = allocation(selector, config)
    by_parent = dict.fromkeys(quality, 0.0)
    for action, probability in zip(actions, beta):
        by_parent[action.parent_id] += probability
    assert by_parent == pytest.approx({pid: value / normalizer for pid, value in quality.items()})
    assert probabilities == beta
    assert all(estimate.eta == 0.0 for estimate in estimates)


@pytest.mark.parametrize("fitness", [(1.0, 2.0, 3.0), (1.0, 2.0, 3.0, 4.0), (1.0, 1.0, 2.0)])
def test_random_reference_prior_matches_base_full_joint_distribution(fitness):
    selector = roots(fitness)
    config = Config()
    archive = {pid: program(pid, node.fitness) for pid, node in selector.nodes.items()}
    quality = weights(archive, config.selection_temperature)
    normalizer = sum(quality.values())
    ids = sorted(archive)
    pools = {pid: [p.id for p in reference_pool(archive, pid)] for pid in ids}
    options = [pools[pid] for pid in ids]
    count = prod(len(pool) for pool in options)
    actual = {}
    for references in product(*options):
        actions, beta, _, probabilities, _ = selector.allocation(
            config, reference_ids=dict(zip(ids, references))
        )
        assert probabilities == beta
        for action, probability in zip(actions, beta):
            actual[action] = actual.get(action, 0.0) + probability / count
    expected = {}
    for pid in ids:
        parent_probability = quality[pid] / normalizer
        expected[Action(pid, "Refine")] = parent_probability * config.refine_weight
        expected[Action(pid, "Explore")] = parent_probability * config.explore_weight
        for reference in pools[pid]:
            expected[Action(pid, "Crossover", reference)] = (
                parent_probability * config.crossover_weight / len(pools[pid])
            )
    assert actual == pytest.approx(expected)


def test_history_window_excludes_old_records():
    selector = roots((100.0, 110.0))
    inject(
        selector,
        11,
        action=Action(1, "Refine"),
        parent_fitness=100.0,
        champion_before=100.0,
        gain=5.0,
        evaluations_before=0,
    )
    selector.evaluations_used = 255
    assert selector.estimate_action(Action(1, "Refine")).eta > 0.0
    selector.evaluations_used = 256
    assert selector.estimate_action(Action(1, "Refine")).eta == 0.0
