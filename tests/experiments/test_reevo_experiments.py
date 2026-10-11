from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.tasks import TASKS

from experiments.reevo import run
from baselines.reevo import ReEvo


@pytest.mark.parametrize("task", TASKS)
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
    assert method._init_sampling_kwargs() == {'temperature': 1.0}
    assert method._sampler.llm.enable_thinking is False
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
    assert payload['method_params']['init_temperature_offset'] == 0.3
    assert "api_key" not in payload["llm"]


@pytest.mark.parametrize('backend,expected', [('deepseek', 1.0), ('server1', 1.3)])
def test_initialization_temperature_matches_recorded_model_adaptation(tmp_path, backend, expected):
    spec = run.make_run_spec(task='tsp_construct', backend=backend, experiments_root=tmp_path)
    directory, name = run.resolve_run_dir(spec)
    run.write_run_config(spec, directory, name)
    method = run.build_method(spec, directory / 'logs')
    try:
        payload = json.loads((directory / 'run_config.json').read_text())
        actual = method._init_sampling_kwargs()['temperature']
        assert actual == expected
        assert actual == payload['llm']['temperature'] + payload['method_params']['init_temperature_offset']
    finally:
        method._evaluation_executor.shutdown()
        method._sampler.llm.close()
