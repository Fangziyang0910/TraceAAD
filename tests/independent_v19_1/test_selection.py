import random

import pytest

from traceaad_v19_1 import Config, Evaluation, Program
from traceaad_v19_1.models import Attempt
from traceaad_v19_1.selection import (
    choose,
    development_stats,
    joint_weights,
    pair_stats,
    qualities,
)


def program(pid, score, parent=0, *, status="success", operator="Refine"):
    evaluation = (
        Evaluation("success", score)
        if status == "success"
        else Evaluation(status, failure=f"{status}: diagnostic {pid}")
    )
    return Program(
        pid,
        f"def solve(x):\n    return {pid}\n",
        f"design_{pid}",
        "Init" if parent == 0 else operator,
        parent,
        None,
        evaluation,
        analysis=f"analysis_{pid}",
    )


def attempt(aid, parent, child, calls=1, *, operator="Refine", completed=True):
    return Attempt(
        aid,
        parent,
        operator,
        None,
        {},
        child,
        [{} for _ in range(calls)],
        completed=completed,
    )


def test_direct_child_quality_is_absolute_and_does_not_use_parent_improvement():
    archive = {1: program(1, 1), 2: program(2, 2), 3: program(3, 3, 1)}
    stats = development_stats(archive, [attempt(1, 1, 3)])

    assert stats[1]["attempts"] == 1
    assert stats[1]["successful_children"] == 1
    assert stats[1]["child_quality_sum"] == pytest.approx(1 / 3)
    assert stats[1]["failed_children"] == 0


def test_failed_attempt_consumes_one_pair_opportunity():
    archive = {1: program(1, 1), 2: program(2, None, 1, status="timeout")}
    stats = pair_stats(archive, [attempt(1, 1, 2, calls=2)])

    assert stats[(1, "Refine")]["attempts"] == 1
    assert stats[(1, "Refine")]["failed_children"] == 1
    assert stats[(1, "Refine")]["model_calls"] == 2
    assert stats[(1, "Refine")]["child_quality_sum"] == 0


def test_joint_state_contains_all_parent_operator_pairs():
    archive = {1: program(1, 1), 2: program(2, 2), 3: program(3, 3)}
    probabilities, diagnostics = joint_weights(archive, [])

    assert set(operator for _, operator in probabilities) == {
        "Refine", "Explore", "Crossover",
    }
    assert len(probabilities) == 9
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert all(diagnostics[pair]["attempts"] == 0 for pair in probabilities)


def test_pair_history_does_not_reduce_another_operator_for_same_parent():
    archive = {1: program(1, 1), 2: program(2, 2), 3: program(3, 3)}
    attempts = [attempt(i, 1, 2, operator="Refine") for i in range(1, 5)]
    _, diagnostics = joint_weights(archive, attempts)

    assert diagnostics[(1, "Refine")]["attempts"] == 4
    assert diagnostics[(1, "Explore")]["attempts"] == 0
    assert diagnostics[(1, "Crossover")]["attempts"] == 0
    assert diagnostics[(1, "Explore")]["uncertainty"] > diagnostics[(1, "Refine")]["uncertainty"]


def test_global_operator_history_changes_joint_operator_mass():
    archive = {
        1: program(1, 1),
        2: program(2, 2),
        3: program(3, 3),
        4: program(4, 4, 1, operator="Crossover"),
        5: program(5, None, 1, status="timeout", operator="Crossover"),
    }
    _, before = joint_weights(archive, [])
    failures = [
        attempt(i, 1, 5, operator="Crossover") for i in range(1, 9)
    ]
    after_probabilities, _ = joint_weights(archive, failures)
    before_probabilities, _ = joint_weights(archive, [])
    before_mass = sum(value for (pid, op), value in before_probabilities.items() if op == "Crossover")
    after_mass = sum(value for (pid, op), value in after_probabilities.items() if op == "Crossover")

    assert before
    assert after_mass < before_mass


def test_choose_records_joint_and_conditional_probabilities():
    archive = {1: program(1, 1), 2: program(2, 2), 3: program(3, 3)}
    parent, operator, reference, selection = choose(
        archive, [], Config(), random.Random(4)
    )

    assert parent.id in archive
    assert operator in {"Refine", "Explore", "Crossover"}
    if operator == "Crossover":
        assert reference is not None and reference.id != parent.id
    assert selection["attempts"] == 0
    assert selection["joint_probability"] > 0
    assert selection["parent_probability"] > 0
    assert selection["operator_probability"] > 0
    assert selection["operator_conditional_probability"] > 0
