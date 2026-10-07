from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.tasks import TASKS

from baselines.funsearch import FunSearch
from experiments.funsearch import run


@pytest.mark.parametrize("task", TASKS)
def test_funsearch_runner_uses_reference_parameters(tmp_path: Path, task) -> None:
    spec = run.make_run_spec(task=task, experiments_root=tmp_path)
    method = run.build_method(spec, tmp_path / "logs")

    assert isinstance(method, FunSearch)
    assert method._max_sample_nums == 1000
    assert method._samples_per_prompt == 4
    config = method._config
    assert (config.num_islands, config.functions_per_prompt) == (10, 2)
    assert config.reset_period == 4 * 60 * 60
    assert (config.cluster_sampling_temperature_init, config.cluster_sampling_temperature_period) == (0.1, 30_000)
    assert len(method._database.islands) == 10


def test_funsearch_run_config_records_parameters(tmp_path: Path) -> None:
    spec = run.make_run_spec(task="online_bin_packing", backend="server1", repeat=2, seed=1,
                             run_name="batch_obp_funsearch_rep2", experiments_root=tmp_path)
    run_dir, run_name = run.resolve_run_dir(spec)
    run.write_run_config(spec, run_dir, run_name)
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))

    assert payload["method"] == "funsearch"
    assert payload["budget"] == 1000
    assert payload["method_params"]["samples_per_prompt"] == 4
    assert payload["method_params"]["num_islands"] == 10
    assert payload["method_params"]["reset_period_seconds"] == 4 * 60 * 60
    assert "api_key" not in payload["llm"]
