import pytest

from experiments.infra.base import TASKS
from experiments.traceaad_bc.launch import build_plan, launch_item


@pytest.mark.parametrize("arm", ["B", "C"])
def test_build_plan_has_fifteen_isolated_runs(arm):
    plan = build_plan("bc_smoke", "bc_smoke", arm, repeats=3, cvrp_last=True)

    assert len(plan) == 15
    assert {row["arm"] for row in plan} == {arm}
    assert {row["task"] for row in plan} == set(TASKS)
    assert [row["task"] for row in plan[:12]].count("cvrp_aco") == 0
    assert all(launch_item(row).extra_args[:2] == ("--arm", arm) for row in plan)
    assert len({row["run_name"] for row in plan}) == 15


def test_build_plan_rejects_unknown_arm():
    with pytest.raises(ValueError):
        build_plan("bc_smoke", "bc_smoke", "A")
