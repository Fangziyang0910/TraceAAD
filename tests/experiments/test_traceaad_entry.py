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
    monkeypatch.setattr(run, "load_evotrace_v3_task", task)
    monkeypatch.setattr(run, "load_evotrace_v4_task", task)
    monkeypatch.setattr(run, "ScheduledModel", Scheduled)
    monkeypatch.setattr(run, "load_environment", lambda _path: None)


def test_retained_dry_runs_preserve_conditions_and_method_result_folders(entry_environment, capsys):
    combinations = {
        "base": [],
        "evotrace.v1": ["tree_population", "reflective_simplification", "shinka_allocation", "multiscale_context"],
        "evotrace.v2": ["tree_population", "dynamic_reflective_simplification", "shinka_allocation", "multiscale_context"],
        "evotrace.v3": ["tree_population", "reflective_tree_simplification", "representative_cover_allocation", "tree_context"],
        "evotrace.v4": ["tree_population", "reflective_elimination", "tree_node_allocation", "traceable_lineage_context"],
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
        if directory == "evotrace_v3":
            assert metadata["selection_rule"] == "representative_rank_direct_sampling"
            assert metadata["allocation"] == {"rank_half_life": 10.0, "explore_references": 6}
        if directory == "evotrace_v4":
            assert metadata["selection_rule"] == "tree_population_rank_direct_sampling"
            assert metadata["allocation"] == {"rank_half_life": 6.0, "explore_references": 6}
            schema = metadata["llm"]["sampling"]["response_format"]["json_schema"]["schema"]
            assert schema["required"] == ["design", "code"]
        mechanisms = combinations[method.replace("_", ".")]
        assert metadata["mechanisms"] == mechanisms
        assert metadata["trajectory_reference_embedding"] == (1 in mechanisms)
        assert ("reference_settings" in metadata) == (1 in mechanisms)
        for key in ("budget", "evaluation_execution", "task_eval",
                    "model_routing", "final_selection"):
            assert metadata[key] == base[key]
        if directory in {"evotrace_v1", "evotrace_v2", "evotrace_v3", "evotrace_v4"}:
            from evotrace_v1 import Config
            from dataclasses import asdict
            expected = asdict(Config())
            expected.update(evaluation_budget=1000, initial_size=5, seed=0, max_tokens=16384, model_timeout=600)
            assert metadata["method_params"] == expected
        else:
            assert metadata["method_params"] == base["method_params"]
        assert metadata["llm"]["model"] == "Qwen3.8-27B"
        assert metadata["llm"]["sampling"]["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize("method", ["v20", "v21", "v21.1", "v21_1", "v21.2", "v21_2", "v22", "v22.1", "v22_1", "v23", "v23.1", "v23_1", "v24", "v21.3", "v21_3", "v22.2", "v22_2", "v23.2", "v23_2", "v24.1", "v24_1"])
def test_removed_methods_are_rejected_before_runtime_setup(method):
    with pytest.raises(SystemExit) as error:
        run.main(["--method", method])
    assert error.value.code == 2


@pytest.mark.parametrize("method", ["evotrace.v1", "evotrace_v1"])
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
    args = Namespace(tasks=tasks, method="base", batch="four", experiment="traceaad_base", repeats=3,
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
    launch_host.main(["--method", "base", "--batch", "four", "--scheduler-socket", "/fake",
                      "--tasks", "tsp_construct", "cvrp_aco", "fssp_gls", "op_aco", "--enqueue"])
    result = json.loads(capsys.readouterr().out)
    assert result["runs"] == 12 and result["status"] == "queued"
    manifest = json.loads((tmp_path / "experiments_result/traceaad_base/batch_four.json").read_text())
    assert len(manifest["tasks"]) == 4
    assert all(row["status"] == "queued" for row in manifest["plan"])
    assert not any(Path(row["run_dir"]).exists() for row in manifest["plan"])


def test_v3_and_v4_parameters_are_recorded_and_other_methods_reject_them(entry_environment, capsys):
    args = ['--task', 'tsp_construct', '--run-name', 'span', '--batch', 'span',
            '--scheduler-socket', '/fake', '--dry-run', '--rank-half-life', '5', '--explore-references', '4',
            '--max-representatives', '16', '--cover-distance', '0.2', '--region-size', '3']
    for method in ('evotrace.v3', 'evotrace.v4'):
        run.main(['--method', method, *args])
        metadata = json.loads(capsys.readouterr().out)
        assert metadata['allocation'] == {'rank_half_life': 5, 'explore_references': 4}
        assert metadata['topology'] == {'max_representatives': 16, 'cover_distance': 0.2, 'region_size': 3}
    with pytest.raises(SystemExit):
        run.main(['--method', 'evotrace.v2', *args])


def test_v3_and_v4_batch_plan_forwards_allocation_and_topology():
    from argparse import Namespace
    from experiments.traceaad.launch_host import build_plan
    for method, exp in (('evotrace.v3', 'evotrace_v3'), ('evotrace.v4', 'evotrace_v4')):
        args = Namespace(tasks=['tsp_construct'], method=method, batch='span',
                         experiment=exp, repeats=1, backend='scheduled', budget=1000,
                         initial_size=5, max_tokens=16384, model_timeout=600,
                         selection_temperature=1., trajectory_reference_embedding=None,
                         scheduler_socket='/fake', seconds_per_instance=10., env_file=None,
                         skip_heldout=False, rank_half_life=5., explore_references=4,
                         max_representatives=16, cover_distance=0.2, region_size=3)
        row, = build_plan(args)
        for flag, value in [('--rank-half-life', '5.0'), ('--explore-references', '4'),
                            ('--max-representatives', '16'), ('--cover-distance', '0.2'), ('--region-size', '3')]:
            assert row['command'][row['command'].index(flag)+1] == value
    args.method = 'evotrace.v2'
    with pytest.raises(ValueError, match='V3/V4 only'):
        build_plan(args)


def test_v3_batch_manifest_records_effective_topology(tmp_path, monkeypatch, capsys):
    from experiments.traceaad import launch_host
    monkeypatch.setattr(launch_host, 'ROOT', tmp_path)
    monkeypatch.setattr(launch_host, 'scheduler_status', lambda _: {'instance_id': 'test'})
    launch_host.main(['--method', 'evotrace.v3', '--batch', 'topology', '--scheduler-socket', '/fake',
                      '--tasks', 'tsp_construct', '--repeats', '1',
                      '--max-representatives', '16', '--cover-distance', '0.2', '--region-size', '3'])
    manifest = json.loads(capsys.readouterr().out)
    assert manifest['topology'] == {'max_representatives': 16, 'cover_distance': 0.2, 'region_size': 3}
    assert not list(tmp_path.rglob('batch_*.json'))
