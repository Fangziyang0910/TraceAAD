import random

import pytest

from traceaad_v19 import Config, Evaluation, Program
from traceaad_v19.models import Attempt
from traceaad_v19.selection import choose, development_stats, qualities, weights


def program(pid, score, parent=0, *, status="success"):
    evaluation = (
        Evaluation("success", score)
        if status == "success"
        else Evaluation(status, failure=f"{status}: diagnostic {pid}")
    )
    return Program(
        pid,
        f"def solve(x):\n    return {pid}\n",
        f"design_{pid}",
        "Init" if parent == 0 else "Refine",
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


def test_failed_attempt_consumes_one_development_opportunity():
    archive = {1: program(1, 1), 2: program(2, None, 1, status="timeout")}
    stats = development_stats(archive, [attempt(1, 1, 2, calls=2)])

    assert stats[1]["attempts"] == 1
    assert stats[1]["failed_children"] == 1
    assert stats[1]["successful_children"] == 0
    assert stats[1]["model_calls"] == 2
    assert stats[1]["child_quality_sum"] == 0


def test_underdeveloped_parent_receives_more_uncertainty():
    archive = {1: program(1, 1), 2: program(2, 2)}
    attempts = [attempt(i, 1, 1) for i in range(1, 9)]
    allocation, diagnostics = weights(archive, attempts, Config())

    assert allocation[1] > 0 and allocation[2] > 0
    assert diagnostics[2]["attempts"] == 0
    assert diagnostics[2]["uncertainty"] > diagnostics[1]["uncertainty"]


def test_development_is_conditioned_on_the_operator():
    archive = {1: program(1, 1), 2: program(2, 2), 3: program(3, 3)}
    attempts = [
        attempt(i, 1, 2, operator="Refine") for i in range(1, 5)
    ] + [
        attempt(5, 2, 3, operator="Explore")
    ]

    refine = development_stats(archive, attempts, operator="Refine")
    explore = development_stats(archive, attempts, operator="Explore")

    assert refine[1]["attempts"] == 4
    assert refine[2]["attempts"] == 0
    assert explore[1]["attempts"] == 0
    assert explore[2]["attempts"] == 1


def test_choose_records_development_statistics_and_keeps_reference_pool():
    archive = {1: program(1, 1), 2: program(2, 2), 3: program(3, 3)}
    parent, operator, reference, selection = choose(archive, [], Config(), random.Random(4))

    assert parent.id in archive
    assert operator in {"Refine", "Explore", "Crossover"}
    if operator == "Crossover":
        assert reference is not None and reference.id != parent.id
    assert selection["attempts"] == 0
    assert selection["successful_children"] == 0
    assert selection["failed_children"] == 0
    assert selection["uncertainty"] > 0
