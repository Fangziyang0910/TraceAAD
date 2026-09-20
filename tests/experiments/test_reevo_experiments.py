from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.infra import base as _common
from experiments.reevo import launch, run
from llm4ad.method.reevo import ReEvo


@pytest.mark.parametrize("task", run.TASKS)
def test_reevo_runner_uses_paper_parameters(tmp_path: Path, task: run.TaskName) -> None:
    spec = run.make_run_spec(task=task, experiments_root=tmp_path)
    method = run.build_method(spec, tmp_path / "logs")

    assert isinstance(method, ReEvo)
    assert spec.max_sample_nums == 1000
    assert spec.pop_size == 10
    assert spec.init_pop_size == 30
    assert spec.mutation_rate == 0.5
    assert method._max_sample_nums == 1000
    assert method._pop_size == 10
    assert method._init_pop_size == 30
    assert method._mutation_rate == 0.5
    method._evaluation_executor.shutdown()


def test_reevo_run_config_records_paper_settings(tmp_path: Path) -> None:
    spec = run.make_run_spec(
        task="cvrp_aco",
        backend="server1",
        repeat=2,
        seed=1,
        run_name="batch_cvrp_reevo_rep2",
        experiments_root=tmp_path,
    )
    run_dir, run_name = run.resolve_run_dir(spec)
    run.write_run_config(spec, run_dir, run_name)
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))

    assert payload["method"] == "reevo"
    assert payload["repeat"] == 2
    assert payload["seed"] == 1
    assert payload["method_params"]["max_sample_nums"] == 1000
    assert payload["method_params"]["population_size"] == 10
    assert payload["method_params"]["init_pop_size"] == 30
    assert payload["method_params"]["mutation_rate"] == 0.5
    assert "api_key" not in payload["llm"]


def test_reevo_launcher_builds_fifteen_runs() -> None:
    args = launch.build_parser().parse_args(
        ["--batch", "20260730_010203", "--dry-run"]
    )
    plan = launch.build_launch_plan(args, module=launch.MODULE, method=launch.METHOD)

    assert len(plan) == 15
    assert {item.task for item in plan} == set(run.TASKS)
    assert {item.repeat for item in plan} == {1, 2, 3}
    assert len({item.session for item in plan}) == 15
    assert len({item.run_dir for item in plan}) == 15
    assert all(item.backend is None for item in plan)


def test_reevo_free_slot_assignment_prefers_remote_backends(monkeypatch) -> None:
    pending = [
        _common.LaunchItem(
            task="tsp_construct",
            repeat=1,
            backend=None,
            session="reevo_tsp_r1",
            run_name="batch_tsp_reevo_rep1",
            run_dir=Path("/tmp/batch_tsp_reevo_rep1"),
            seed=0,
            module=launch.MODULE,
        ),
        _common.LaunchItem(
            task="cvrp_aco",
            repeat=1,
            backend=None,
            session="reevo_cvrp_r1",
            run_name="batch_cvrp_reevo_rep1",
            run_dir=Path("/tmp/batch_cvrp_reevo_rep1"),
            seed=0,
            module=launch.MODULE,
        ),
        _common.LaunchItem(
            task="op_aco",
            repeat=1,
            backend=None,
            session="reevo_op_r1",
            run_name="batch_op_reevo_rep1",
            run_dir=Path("/tmp/batch_op_reevo_rep1"),
            seed=0,
            module=launch.MODULE,
        ),
    ]
    monkeypatch.setattr(
        _common,
        "free_slots",
        lambda: {"server3": 2, "server3b": 2, "local": 0},
    )
    assigned = _common.assign_backends(pending)
    assert [item.backend for item in assigned] == ["server3", "server3b", "server3"]
