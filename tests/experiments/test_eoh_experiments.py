from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.infra import base as _common
from experiments.eoh import launch, run
from llm4ad.method.eoh import EoH


@pytest.mark.parametrize("task", run.TASKS)
def test_eoh_runner_uses_paper_parameters(tmp_path: Path, task: run.TaskName) -> None:
    spec = run.make_run_spec(task=task, experiments_root=tmp_path)
    method = run.build_method(spec, tmp_path / "logs")

    assert isinstance(method, EoH)
    assert spec.generations == 20
    assert spec.effective_pop_size == (
        20 if task == "online_bin_packing" else 10
    )
    assert spec.effective_budget == 1000
    assert method._selection_num == 5
    assert method._operators == ["e1", "e2", "m1", "m2"]
    assert method._operator_weights == [1.0, 1.0, 1.0, 1.0]
    assert method._use_m3_operator is False
    assert method._initial_sample_nums_max == 2 * spec.effective_pop_size
    method._evaluation_executor.shutdown()


def test_eoh_run_config_records_paper_adaptation(tmp_path: Path) -> None:
    spec = run.make_run_spec(
        task="online_bin_packing",
        backend="server1",
        repeat=2,
        seed=1,
        run_name="batch_obp_eoh_rep2",
        experiments_root=tmp_path,
    )
    run_dir, run_name = run.resolve_run_dir(spec)
    run.write_run_config(spec, run_dir, run_name)
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))

    assert payload["method"] == "eoh"
    assert payload["repeat"] == 2
    assert payload["seed"] == 1
    assert payload["method_params"]["paper_generations"] == 20
    assert payload["method_params"]["max_sample_nums"] == 1000
    assert payload["method_params"]["population_size"] == 20
    assert payload["method_params"]["selection_num"] == 5
    assert payload["method_params"]["operators"] == ["e1", "e2", "m1", "m2"]
    assert payload["method_params"]["m3_enabled"] is False
    assert "api_key" not in payload["llm"]


def test_eoh_launcher_builds_fifteen_runs() -> None:
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


def test_eoh_free_slot_assignment_prefers_remote_backends(monkeypatch) -> None:
    pending = [
        _common.LaunchItem(
            task="tsp_construct",
            repeat=1,
            backend=None,
            session="eoh_tsp_r1",
            run_name="batch_tsp_eoh_rep1",
            run_dir=Path("/tmp/batch_tsp_eoh_rep1"),
            seed=0,
            module=launch.MODULE,
        ),
        _common.LaunchItem(
            task="cvrp_aco",
            repeat=1,
            backend=None,
            session="eoh_cvrp_r1",
            run_name="batch_cvrp_eoh_rep1",
            run_dir=Path("/tmp/batch_cvrp_eoh_rep1"),
            seed=0,
            module=launch.MODULE,
        ),
        _common.LaunchItem(
            task="op_aco",
            repeat=1,
            backend=None,
            session="eoh_op_r1",
            run_name="batch_op_eoh_rep1",
            run_dir=Path("/tmp/batch_op_eoh_rep1"),
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
