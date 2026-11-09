"""Retained methods share evaluation conditions and route to their own result folders."""

import json

import pytest

from experiments.traceaad import run
from traceaad_base import Task


def test_retained_dry_runs_preserve_conditions_and_method_result_folders(monkeypatch, capsys):
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
    monkeypatch.setattr(run, "ScheduledModel", Scheduled)
    monkeypatch.setattr(run, "load_environment", lambda _path: None)
    arguments = ["--task", "tsp_construct", "--run-name", "test", "--batch", "test",
                 "--scheduler-socket", "/fake", "--dry-run"]
    run.main(arguments)
    base = json.loads(capsys.readouterr().out)
    assert base["method"] == base["experiment"] == "traceaad_base"
    assert base["evaluation_execution"]["timeout_seconds"] == 160
    for method, (directory, _package) in run.METHODS.items():
        run.main([*arguments, "--method", method])
        metadata = json.loads(capsys.readouterr().out)
        assert metadata["method"] == metadata["experiment"] == directory
        for key in ("budget", "method_params", "evaluation_execution", "task_eval",
                    "model_routing", "final_selection"):
            assert metadata[key] == base[key]
        assert metadata["llm"]["model"] == "Qwen3.8-27B"
        assert metadata["llm"]["sampling"]["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize("method", ["v20", "v22", "v23"])
def test_removed_methods_are_rejected_before_runtime_setup(method):
    with pytest.raises(SystemExit) as error:
        run.main(["--method", method])
    assert error.value.code == 2
