from collections import Counter

from experiments.infra.base import TASKS
from experiments.traceaad_v10_11.launch import allocate, build_plan, launch_item


def test_two_repeat_no_history_plan_preserves_run_settings():
    plan = build_plan("no_history", "nh", repeats=2, traj_gens=0)
    assert Counter(row["task"] for row in plan) == {task: 2 for task in TASKS}
    assert {(row["repeat"], row["seed"]) for row in plan} == {(1, 0), (2, 1)}
    for row in plan:
        assert launch_item(row).extra_args == ("--traj-gens", "0")
    assert len(build_plan("default", "default")) == 15
    assert launch_item(build_plan("default", "default")[0]).extra_args == ("--traj-gens", "8")


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
