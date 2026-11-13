"""Retained methods share evaluation conditions and route to their own result folders."""

import json
from pathlib import Path

import pytest

from experiments.traceaad import run
from traceaad_base import Task


@pytest.fixture
def entry_environment(monkeypatch):
    def task(**_kwargs):
        return Task(
            "minimize", "def solve(x): return x", lambda _source, fn: fn(1),
            time_limit=10,
            metadata={"n_instance": 16, "settings": {"timeout_seconds": 10}},
        )

    class Scheduled:
        model = "Qwen3.8-27B"

        def __init__(self, path, _label):
            self.path = path
            self.scheduler = {"endpoints": []}

    monkeypatch.setattr(run, "load_task", task)
    monkeypatch.setattr(run, "load_evotrace_task", task)
    monkeypatch.setattr(run, "load_evotrace_v2_task", task)
    monkeypatch.setattr(run, "ScheduledModel", Scheduled)
    monkeypatch.setattr(run, "load_environment", lambda _path: None)


def test_retained_dry_runs_preserve_conditions_and_method_result_folders(entry_environment, capsys):
    combinations = {
        "base": [],
        "evotrace.v1": ["tree_population", "reflective_simplification", "shinka_allocation", "multiscale_context"],
        "evotrace.v2": ["tree_population", "dynamic_reflective_simplification", "shinka_allocation", "multiscale_context"],
        "v21.3": [2],
        "v22.2": [1, 2],
        "v23.2": [1],
        "v24.1": [2, 3],
    }
    assert set(run.METHODS) == set(combinations) | {m.replace(".", "_") for m in combinations}
    arguments = ["--task", "tsp_construct", "--run-name", "test", "--batch", "test",
                 "--scheduler-socket", "/fake", "--dry-run", "--selection-temperature", "0.1"]
    run.main(arguments)
    base = json.loads(capsys.readouterr().out)
    assert base["method"] == base["experiment"] == "traceaad_base"
    assert base["evaluation_execution"]["timeout_seconds"] == 160
    assert base["selection_rule"] == "boltzmann_fitness_v1"
    assert base["method_params"]["selection_temperature"] == 0.1
    assert base["llm"]["temperature"] == 1.0
    for method, (directory, _search_type) in run.METHODS.items():
        run.main([*arguments, "--method", method])
        metadata = json.loads(capsys.readouterr().out)
        assert metadata["method"] == metadata["experiment"] == directory
        mechanisms = combinations[method.replace("_", ".")]
        assert metadata["mechanisms"] == mechanisms
        assert metadata["trajectory_reference_embedding"] == (1 in mechanisms)
        assert ("reference_settings" in metadata) == (1 in mechanisms)
        for key in ("budget", "evaluation_execution", "task_eval",
                    "model_routing", "final_selection"):
            assert metadata[key] == base[key]
        if directory in {"evotrace_v1", "evotrace_v2"}:
            from evotrace_v1 import Config
            from dataclasses import asdict
            expected = asdict(Config())
            expected.update(evaluation_budget=1000, initial_size=5, seed=0, max_tokens=16384, model_timeout=600)
            assert metadata["method_params"] == expected
        else:
            assert metadata["method_params"] == base["method_params"]
        assert metadata["llm"]["model"] == "Qwen3.8-27B"
        assert metadata["llm"]["sampling"]["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize("method", ["v20", "v21", "v21.1", "v21_1", "v21.2", "v21_2", "v22", "v22.1", "v22_1", "v23", "v23.1", "v23_1", "v24"])
def test_removed_methods_are_rejected_before_runtime_setup(method):
    with pytest.raises(SystemExit) as error:
        run.main(["--method", method])
    assert error.value.code == 2


def test_trajectory_embedding_dry_run_records_policy_without_embedding(monkeypatch, capsys):
    from experiments.deltaevolve import embedding

    class Encoder:
        settings = {"model": "test-embedding"}

        def __call__(self, text):
            raise AssertionError("Dry-run must not make embedding requests")

    class Scheduled:
        model = "test-model"

        def __init__(self, path, label):
            self.path, self.scheduler = path, {"endpoints": []}

    monkeypatch.setattr(embedding, "CodeEmbedding", Encoder)
    monkeypatch.setattr(run, "ScheduledModel", Scheduled)
    monkeypatch.setattr(run, "load_environment", lambda _path: None)
    monkeypatch.setattr(run, "load_task", lambda **kw: Task(
        "minimize", "def solve(x): return x", lambda src, fn: fn(0), 10,
        metadata={"n_instance": 16, "settings": {"timeout_seconds": 10}},
    ))
    arguments = ["--task", "tsp_construct", "--run-name", "test", "--batch", "test",
                 "--scheduler-socket", "/fake", "--dry-run", "--trajectory-reference-embedding"]
    run.main([*arguments, "--method", "v23.2"])
    metadata = json.loads(capsys.readouterr().out)
    assert metadata["reference_settings"]["rule"] == "trajectory_embedding_diverse_v1"
    assert metadata["method"] == metadata["experiment"] == "traceaad_v23_2"
    assert metadata["reference_settings"]["embedding"]["model"] == "test-embedding"
    with pytest.raises(SystemExit) as error:
        run.main([*arguments, "--method", "base"])
    assert error.value.code == 2


@pytest.mark.parametrize("method", ["v22.2", "v23.2"])
@pytest.mark.parametrize("requested,enabled", [(None, True), (True, True), (False, False)])
def test_batch_plan_forwards_trajectory_embedding_switch(method, requested, enabled):
    from argparse import Namespace
    from experiments.traceaad.launch_host import build_plan

    args = Namespace(tasks=None, method=method, batch="test", experiment="traceaad_" + method.replace(".", "_"), repeats=1,
                     backend="scheduled", budget=1000, initial_size=5, max_tokens=16384,
                     model_timeout=600, selection_temperature=1.0, trajectory_reference_embedding=requested,
                     scheduler_socket="/fake", seconds_per_instance=10.0, env_file=None, skip_heldout=False)
    for row in build_plan(args):
        assert ("--trajectory-reference-embedding" in row["command"]) == enabled
        assert ("--no-trajectory-reference-embedding" in row["command"]) != enabled


@pytest.mark.parametrize("method", ["v22.2", "v23.2"])
def test_reference_ablation_is_explicit_and_recorded(entry_environment, capsys, method):
    run.main(["--method", method, "--task", "tsp_construct", "--run-name", "ablation",
              "--batch", "ablation", "--scheduler-socket", "/fake", "--dry-run",
              "--no-trajectory-reference-embedding"])
    metadata = json.loads(capsys.readouterr().out)
    assert metadata["trajectory_reference_embedding"] is False
    assert "reference_settings" not in metadata
    assert 1 in metadata["mechanisms"]


@pytest.mark.parametrize("method", ["v21.3", "v21_3", "evotrace.v1", "evotrace_v1"])
def test_reference_source_control_is_recorded_for_supported_methods(entry_environment, capsys, method):
    run.main(["--method", method, "--task", "tsp_construct", "--run-name", "source_control",
              "--batch", "source_control", "--scheduler-socket", "/fake", "--dry-run", "--reference-code"])
    metadata = json.loads(capsys.readouterr().out)
    assert metadata["reference_code"] is True
    expected_protocol = run.METHODS[method][1].generation_protocol
    if run.METHODS[method][0] != "evotrace_v1":
        expected_protocol += "+reference_code"
    assert metadata["generation_protocol"] == expected_protocol


def test_reference_source_control_rejects_base_before_runtime_setup():
    with pytest.raises(SystemExit) as error:
        run.main(["--method", "base", "--task", "tsp_construct", "--run-name", "test",
                  "--batch", "test", "--reference-code"])
    assert error.value.code == 2


def test_four_task_plan_has_exactly_twelve_routes():
    from argparse import Namespace
    from experiments.traceaad.launch_host import build_plan

    tasks = ["tsp_construct", "cvrp_aco", "fssp_gls", "op_aco"]
    args = Namespace(tasks=tasks, method="v21.3", batch="four", experiment="traceaad_v21_3", repeats=3,
                     backend="scheduled", budget=1000, initial_size=5, max_tokens=16384,
                     model_timeout=600, selection_temperature=1.0, trajectory_reference_embedding=None,
                     scheduler_socket="/fake", seconds_per_instance=10.0, env_file=None, skip_heldout=False)
    plan = build_plan(args)
    assert len(plan) == 12
    assert {(r["task"], r["seed"]) for r in plan} == {(task, seed) for task in tasks for seed in range(3)}
    args.tasks = ["tsp_construct", "tsp_construct"]
    with pytest.raises(ValueError, match="distinct"):
        build_plan(args)


def test_enqueue_saves_the_subset_without_starting_workers(tmp_path, monkeypatch, capsys):
    from experiments.traceaad import launch_host

    monkeypatch.setattr(launch_host, "ROOT", tmp_path)
    monkeypatch.setattr(launch_host, "scheduler_status", lambda _path: {"instance_id": "test"})
    monkeypatch.setattr(launch_host, "is_session_alive", lambda _name: False)
    monkeypatch.setattr(launch_host, "launch_run", lambda _row: pytest.fail("enqueue must not start workers"))
    launch_host.main(["--method", "v21.3", "--batch", "four", "--scheduler-socket", "/fake",
                      "--tasks", "tsp_construct", "cvrp_aco", "fssp_gls", "op_aco", "--enqueue"])
    result = json.loads(capsys.readouterr().out)
    assert result["runs"] == 12 and result["status"] == "queued"
    manifest = json.loads((tmp_path / "experiments_result/traceaad_v21_3/batch_four.json").read_text())
    assert len(manifest["tasks"]) == 4
    assert all(row["status"] == "queued" for row in manifest["plan"])
    assert not any(Path(row["run_dir"]).exists() for row in manifest["plan"])
