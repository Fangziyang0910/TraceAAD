from collections import Counter

from experiments.traceaad_v10_13.launch import (
    BACKEND_POOL,
    TARGET_DISTRIBUTION,
    build_plan,
    validate_plan,
)
from experiments.infra.base import TASKS


def test_v1013_plan_has_four_repeats_and_confirmed_three_pool_distribution():
    plan = build_plan("test_v1013", "test_v1013")
    validate_plan(plan)
    assert len(plan) == 20
    assert Counter(row["backend"] for row in plan) == Counter(TARGET_DISTRIBUTION)
    assert set(row["backend"] for row in plan) == set(BACKEND_POOL)
    assert {(row["task"], row["repeat"]) for row in plan} == {
        (task, repeat) for task in TASKS for repeat in range(1, 5)
    }
    assert all(row["session"].startswith("test_v1013_") for row in plan)
