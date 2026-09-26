"""One launch plan for all comparable methods."""

import pytest

from experiments import launch
from experiments.infra.base import REPO_ROOT, RESULTS_ROOT, TASKS, assign_backends
from experiments.eoh import run as eoh_run


def test_results_root_has_no_runs_layer():
    assert RESULTS_ROOT == REPO_ROOT / "experiments_result"


@pytest.mark.parametrize("method", launch.METHODS)
def test_baseline_plan_covers_five_tasks_and_three_repeats(method):
    args = launch.build_parser().parse_args([
        "--method", method, "--batch", "test_batch", "--dry-run",
    ])
    plan = launch.build_plan(args)

    assert {(item.task, item.repeat) for item in plan} == {
        (task, repeat) for task in TASKS for repeat in (1, 2, 3)
    }
    assert len({item.run_dir for item in plan}) == 15
    assert all(item.run_dir == RESULTS_ROOT / method / item.task / item.run_name
               for item in plan)
    assert all(item.module == f"experiments.{method}.run" for item in plan)


def test_plan_filters_tasks_and_passes_method_options():
    args = launch.build_parser().parse_args([
        "--method", "eoh", "--batch", "probe", "--tasks", "tsp_construct",
        "--repeats", "2", "--run-arg=--budget=20",
    ])
    plan = launch.build_plan(args)

    assert len(plan) == 2
    assert {item.task for item in plan} == {"tsp_construct"}
    assert all(item.command()[-1] == "--budget=20" for item in
               (item.with_backend("server3") for item in plan))


def test_scheduler_assigns_available_endpoints(monkeypatch):
    args = launch.build_parser().parse_args([
        "--method", "reevo", "--batch", "test_batch", "--tasks", "tsp_construct",
    ])
    plan = launch.build_plan(args)
    monkeypatch.setattr(
        "experiments.infra.base.free_slots",
        lambda: {"server3": 2, "server3b": 1, "local": 0},
    )

    assigned = assign_backends(plan)

    assert [item.backend for item in assigned] == ["server3", "server3", "server3b"]


def test_launcher_and_single_run_agree_on_output_path():
    args = launch.build_parser().parse_args([
        "--method", "eoh", "--batch", "batch", "--tasks", "tsp_construct",
        "--repeats", "1",
    ])
    item = launch.build_plan(args)[0]
    spec = eoh_run.make_run_spec(task=item.task, run_name=item.run_name)

    assert spec.experiment_root / spec.run_name == item.run_dir
