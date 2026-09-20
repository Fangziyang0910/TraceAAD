from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.infra import base as _common
from experiments.shinka_evo import launch, run
from llm4ad.method.shinka_evo import ShinkaEvo


@pytest.mark.parametrize("task", run.TASKS)
def test_shinka_runner_uses_paper_parameters(tmp_path: Path, task: run.TaskName) -> None:
    spec = run.make_run_spec(task=task, experiments_root=tmp_path)
    method = run.build_method(spec, tmp_path / "logs")

    assert isinstance(method, ShinkaEvo)
    assert spec.max_sample_nums == 1000
    assert spec.num_generations == 1000
    assert spec.archive_size == 40
    assert spec.num_islands == 2
    assert spec.num_archive_inspirations == 4
    assert spec.num_top_k_inspirations == 2
    assert spec.migration_rate == 0.0
    assert list(spec.patch_type_probs) == [0.45, 0.45, 0.1]
    assert method._max_sample_nums == 1000
    assert method._num_generations == 1000
    assert method._meta_llm is not None
    assert method._novelty_llm is None
    assert method._embedding_fn is None
    for llm in method._all_llms():
        llm.close()


def test_shinka_run_config_records_paper_settings(tmp_path: Path) -> None:
    spec = run.make_run_spec(
        task="cvrp_aco",
        backend="server1",
        repeat=2,
        seed=1,
        run_name="batch_cvrp_shinka_rep2",
        experiments_root=tmp_path,
    )
    run_dir, run_name = run.resolve_run_dir(spec)
    run.write_run_config(spec, run_dir, run_name)
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))

    assert payload["method"] == "shinka_evo"
    assert payload["repeat"] == 2
    assert payload["method_params"]["max_sample_nums"] == 1000
    assert payload["method_params"]["num_generations"] == 1000
    assert payload["method_params"]["archive_size"] == 40
    assert payload["method_params"]["num_archive_inspirations"] == 4
    assert payload["method_params"]["novelty_rejection_enabled"] is False
    assert "api_key" not in payload["llm"]


def test_shinka_launcher_builds_fifteen_runs() -> None:
    args = launch.build_parser().parse_args(
        ["--batch", "20260730_010203", "--dry-run"]
    )
    plan = launch.build_launch_plan(args, module=launch.MODULE, method=launch.METHOD)

    assert len(plan) == 15
    assert {item.task for item in plan} == set(run.TASKS)
    assert {item.repeat for item in plan} == {1, 2, 3}
    assert all(item.backend is None for item in plan)


def test_shinka_free_slot_assignment_prefers_remote_backends(monkeypatch) -> None:
    pending = [
        _common.LaunchItem(
            task="tsp_construct",
            repeat=1,
            backend=None,
            session="shinka_tsp_r1",
            run_name="batch_tsp_shinka_rep1",
            run_dir=Path("/tmp/batch_tsp_shinka_rep1"),
            seed=0,
            module=launch.MODULE,
        ),
        _common.LaunchItem(
            task="cvrp_aco",
            repeat=1,
            backend=None,
            session="shinka_cvrp_r1",
            run_name="batch_cvrp_shinka_rep1",
            run_dir=Path("/tmp/batch_cvrp_shinka_rep1"),
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
    assert [item.backend for item in assigned] == ["server3", "server3b"]
