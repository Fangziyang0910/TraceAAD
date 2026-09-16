from collections import Counter

import pytest

from experiments.infra.base import TASKS
from experiments.traceaad_v10_11.launch import allocate, build_plan, launch_item, main as launch_main


def test_two_repeat_no_history_plan_preserves_run_settings():
    plan = build_plan("no_history", "nh", repeats=2, traj_gens=0)
    assert Counter(row["task"] for row in plan) == {task: 2 for task in TASKS}
    assert {(row["repeat"], row["seed"]) for row in plan} == {(1, 0), (2, 1)}
    for row in plan:
        assert launch_item(row).extra_args == ("--traj-gens", "0")
    assert len(build_plan("default", "default")) == 15
    assert launch_item(build_plan("default", "default")[0]).extra_args == ("--traj-gens", "8")


def test_rand_context_plan_swaps_traj_gens_for_reference_flags():
    plan = build_plan("rand_ctx", "rc", rand_context=True, cvrp_last=True, n_references=8)
    assert len(plan) == 15
    assert [row["task"] for row in plan[:12]].count("cvrp_aco") == 0
    for row in plan:
        assert row["traj_gens"] == 0 and row["rand_context"] is True
        assert row["n_references"] == 8
        assert launch_item(row).extra_args == ("--rand-context", "--n-references", "8")
    plan = build_plan("default", "default")
    assert all("rand_context" not in row for row in plan)
    assert launch_item(plan[0]).extra_args == ("--traj-gens", "8")


@pytest.mark.parametrize("argv", [
    ["--batch", "x", "--rand-context", "--history-code"],
    ["--batch", "x", "--rand-context", "--traj-gens", "4"],
    ["--batch", "x", "--n-references", "5"],
    ["--batch", "x", "--rand-context", "--n-references", "0"],
])
def test_launch_rejects_conflicting_ablation_flags(argv):
    with pytest.raises(SystemExit):
        launch_main(argv)


def test_idea_code_plan_fills_non_cvrp_slots_before_cpu_heavy_cvrp():
    plan = build_plan("idea_code", "ic", history_code=True, cvrp_last=True)
    assert len(plan) == 15
    assert [row["task"] for row in plan[:12]].count("cvrp_aco") == 0
    assert [row["task"] for row in plan[12:]] == ["cvrp_aco"] * 3
    assert all("--history-code" in launch_item(row).extra_args for row in plan)
    available = {"local": 0, "server1": 2, "server3": 2, "server3b": 2}
    first = allocate(plan, available, tuple(available), defer_cvrp=True)
    assert len(first) == 6
    assert all(row["task"] != "cvrp_aco" for row, _ in first)
    for row, _ in first:
        row["status"] = "running"
    next_up = allocate(plan, {"server3b": 1}, ("server3b",), defer_cvrp=True)
    assert len(next_up) == 1 and next_up[0][0]["task"] != "cvrp_aco"
    for row in plan:
        if row["task"] != "cvrp_aco":
            row["status"] = "running"
    assert allocate(plan, {"server3b": 1}, ("server3b",), defer_cvrp=True) == []
    assert allocate(plan, {"server3b": 1}, ("server3b",))[0][0]["task"] == "cvrp_aco"
