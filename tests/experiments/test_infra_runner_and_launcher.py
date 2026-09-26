"""Unit tests for standard experiment runner and launcher harnesses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.infra.launcher import get_summary_status
from experiments.infra.runner import (
    add_common_run_args,
    resolve_resumable_run_dir,
    run_baseline_experiment,
    setup_experiment_run,
)


def test_add_common_run_args_and_parser():
    parser = argparse.ArgumentParser()
    add_common_run_args(parser)
    args = parser.parse_args(["--task", "tsp_construct", "--backend", "local", "--seed", "42"])
    assert args.task == "tsp_construct"
    assert args.backend == "local"
    assert args.seed == 42
    assert args.budget == 1000


def test_resolve_resumable_run_dir(tmp_path: Path):
    task_root = tmp_path / "results" / "tsp_construct"
    run_dir, run_name, resumed = resolve_resumable_run_dir(task_root, "run_01", "tree_state.json")
    assert not resumed
    assert run_dir.exists()
    assert run_name == "run_01"

    # With resume file present
    (run_dir / "tree_state.json").write_text("{}", encoding="utf-8")
    run_dir2, run_name2, resumed2 = resolve_resumable_run_dir(task_root, "run_01", "tree_state.json")
    assert resumed2
    assert run_dir2 == run_dir


def test_setup_experiment_run_writes_valid_config(tmp_path: Path):
    parser = argparse.ArgumentParser()
    add_common_run_args(parser)
    args = parser.parse_args([
        "--task", "online_bin_packing",
        "--backend", "local",
        "--seed", "3",
        "--run-name", "test_obp_run",
    ])

    ctx = setup_experiment_run(
        args,
        method="test_method",
        results_root=tmp_path,
        resume_file="tree_state.json",
        method_params={"custom_param": 123},
        budget_basis="1000 budget test",
    )

    assert ctx.task == "online_bin_packing"
    assert ctx.seed == 3
    assert not ctx.resumed

    config_path = ctx.run_dir / "run_config.json"
    assert config_path.exists()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["task"] == "online_bin_packing"
    assert payload["method"] == "test_method"
    assert payload["method_params"]["custom_param"] == 123
    assert payload["method_params"]["budget_basis"] == "1000 budget test"


def test_launcher_status_and_session_naming(tmp_path: Path):
    run_dir = tmp_path / "run"
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True)

    assert get_summary_status(run_dir) is None

    (log_dir / "run_summary.json").write_text(json.dumps({"status": "finished"}), encoding="utf-8")
    assert get_summary_status(run_dir) == "finished"



def test_baseline_lifecycle_runs_once_and_writes_config(tmp_path, monkeypatch):
    from experiments.infra import runner

    calls = []
    spec = SimpleNamespace(experiment_root=tmp_path / "eoh" / "tsp_construct",
                           run_name="rep1", model="model", base_url="local")
    monkeypatch.setattr(runner, "run_in_tmux_log",
                        lambda run_dir, log_dir, header, body: body())
    def write_config(spec, run_dir, run_name):
        (run_dir / "run_config.json").write_text(json.dumps({"run_name": run_name}))
    def build_method(spec, log_dir):
        return SimpleNamespace(run=lambda: calls.append(log_dir))

    run_dir = run_baseline_experiment(spec, write_config=write_config,
                                      build_method=build_method, description="eoh")
    assert run_dir == tmp_path / "eoh" / "tsp_construct" / "rep1"
    assert json.loads((run_dir / "run_config.json").read_text())["run_name"] == "rep1"
    assert calls == [run_dir / "logs"]
