"""Offline checks for V18.1 configuration, launching, resume and generation budgets."""

from dataclasses import asdict
import json

import pytest

from benchmarks.tasks import CO_TASKS, FUNCTION_LIMITS, INSTANCE_LIMITS
from experiments.traceaad_v18_1 import runner
from experiments.traceaad_v18_1 import launch_host as search_host_launch, run as search_run
from experiments.traceaad_v18_1.launch_host import main as launch
from experiments.traceaad_v18_1.run import main
from tests.support import TinyEvaluation, TokenLLM, response
from traceaad.base.storage import read_json, rows, write_json
from traceaad.v18_1 import Config, TraceAADV181


@pytest.fixture(autouse=True)
def offline_entry(monkeypatch, tmp_path):
    def unexpected_connection(*args, **kwargs):
        pytest.fail("entry tests must not connect to a model service or launch processes")

    monkeypatch.setattr(search_run, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(search_host_launch, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(runner, "build_llm_client", unexpected_connection)
    monkeypatch.setattr(search_host_launch, "launch_plan", unexpected_connection)


@pytest.mark.parametrize("task", CO_TASKS)
def test_v18_1_dry_run_uses_shared_tasks_and_generation_budget(task, capsys, tmp_path):
    main(["--task", task, "--eval-workers", "2", "--dry-run"])
    preview = json.loads(capsys.readouterr().out)
    config = preview["config"]
    assert preview["method"] == TraceAADV181.METHOD == "traceaad_v18_1"
    assert config["budget"] == 1000
    assert config["output_tokens"] == 8192
    assert config["same_age"] is True
    assert config["age_fraction"] == 0.25
    assert config["protection_attempts"] == 12
    assert config["max_refinement_age"] == 8
    assert config["birth_output_tokens"] == 2048
    assert config["birth_context_tokens"] == 24000
    assert "max_model_calls" not in config
    assert "lds" not in config
    assert preview["budget_axis"] == "候选尝试"
    assert "BirthCheck" in preview["budget_basis"]
    assert "do not consume the generation budget" in preview["budget_basis"]
    assert "costs are recorded" in preview["budget_basis"]
    assert preview["evaluation_execution"] == {
        "timeout_scope": "instance",
        "function_budget_scope": "mean_instance",
        "timeout_seconds": INSTANCE_LIMITS[task],
        "function_seconds": FUNCTION_LIMITS[task],
        "n_workers": 2,
    }
    assert preview["final_selection"] == "training"
    assert "separate held-out" in preview["test"]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("same_age", [True, False])
def test_v18_1_host_preview_roundtrips_every_policy_option(same_age, capsys, tmp_path):
    launch([
        "--batch", "v18_1_preview", "--suite", "co6", "--backends", "server3",
        "--budget", "11", "--eval-workers", "2", "--eval-timeout-seconds", "73",
        "--roots", "3", "--init-attempt-limit", "9", "--history-depth", "5",
        "--attempts-shown", "4", "--progress-shown", "6", "--age-fraction", "0.375",
        "--protection-attempts", "7", "--max-refinement-age", "4",
        "--birth-output-tokens", "1024", "--birth-context-tokens", "16000",
        "--output-tokens", "4096", "--same-age" if same_age else "--no-same-age",
    ])
    batch = json.loads(capsys.readouterr().out)
    assert batch["method"] == batch["experiment"] == "traceaad_v18_1"
    assert len(batch["plan"]) == 18
    assert batch["budget_per_run"] == 11
    assert batch["budget_axis"] == "候选尝试"
    assert batch["search_policy"]["same_age"] is same_age
    assert batch["search_policy"]["age_fraction"] == 0.375
    assert batch["search_policy"]["birth_output_tokens"] == 1024
    assert batch["llm"]["server3"]["max_tokens"] == 4096
    for item in batch["plan"]:
        assert item["command"][1:3] == ["-m", "experiments.traceaad_v18_1.run"]
        assert item["seed"] == item["repeat"] - 1
        assert "traceaad_v18_1" in item["run_dir"]
        assert "--birth-context-tokens=16000" in item["command"]
        main([*item["command"][3:], "--dry-run"])
        single = json.loads(capsys.readouterr().out)
        assert single["config"] == {**batch["search_policy"], "seed": item["seed"]}
        assert single["budget_basis"] == batch["budget_basis"]
    assert not list(tmp_path.iterdir())


def test_v18_1_host_passes_scheduled_plan_to_launcher(monkeypatch, tmp_path, capsys):
    socket = str(tmp_path / "scheduler.sock")
    scheduler = {
        "socket": socket,
        "instance_id": "offline",
        "cpu": {"capacity": 2},
        "endpoints": [{"base_url": "http://localhost:8000/v1", "model": "offline-model"}],
    }
    monkeypatch.setattr(search_host_launch, "scheduler_status", lambda path: scheduler)
    launched = []
    monkeypatch.setattr(search_host_launch, "launch_plan", lambda *a, **kw: launched.append((a, kw)))
    launch([
        "--batch", "v18_1_launch", "--tasks", "tsp_construct", "--backends", "server3",
        "--repeat-ids", "2", "--scheduler-socket", socket, "--launch",
    ])
    assert len(launched) == 1
    (path, batch), kwargs = launched[0]
    assert path == tmp_path / "traceaad_v18_1" / "batch_v18_1_launch.json"
    assert kwargs == {"min_context": 32768}
    assert batch["resource_scheduler"] == scheduler
    assert batch["search_policy"]["budget"] == 1000
    assert batch["search_policy"]["output_tokens"] == 8192
    assert batch["budget_axis"] == "候选尝试"
    assert "BirthCheck" in batch["budget_basis"]
    assert len(batch["plan"]) == 1
    item = batch["plan"][0]
    assert item["backend"] == "scheduled"
    assert item["repeat"] == 2 and item["seed"] == 1
    assert item["command"][1:3] == ("-m", "experiments.traceaad_v18_1.run")
    assert "--eval-workers=2" in item["command"]
    assert socket in item["command"]
    assert not capsys.readouterr().out
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("field,value", [
    ("same_age", False), ("age_fraction", 0.5), ("protection_attempts", 3),
    ("max_refinement_age", 4), ("birth_output_tokens", 1024),
    ("birth_context_tokens", 16000), ("budget", 999), ("output_tokens", 4096),
    ("roots", 2), ("method", "traceaad_v18"),
])
def test_v18_1_resume_rejects_changed_config_before_setup(
    field, value, dry_run, tmp_path, monkeypatch
):
    run_dir = tmp_path / "traceaad_v18_1" / "tsp_construct" / "resume"
    params = asdict(Config(eval_timeout_seconds=INSTANCE_LIMITS["tsp_construct"]))
    saved = {"method": "traceaad_v18_1", "method_params": params}
    if field == "method":
        saved[field] = value
    else:
        params[field] = value
    write_json(run_dir / "run_config.json", saved)
    (run_dir / "events.jsonl").touch()
    before = (run_dir / "run_config.json").read_bytes()
    monkeypatch.setattr(search_run, "setup_experiment_run", lambda *a, **kw: pytest.fail("setup called"))
    args = ["--task", "tsp_construct", "--run-name", "resume"]
    with pytest.raises(ValueError, match="resume configuration differs"):
        main(args + (["--dry-run"] if dry_run else []))
    assert (run_dir / "run_config.json").read_bytes() == before


def test_v18_1_entry_records_reply_budget_and_extra_birth_costs_then_resumes(
    tmp_path, monkeypatch, capsys
):
    class ScriptedLLM(TokenLLM):
        closed = False

        def draw_sample_with_details(self, prompt, **kwargs):
            details = super().draw_sample_with_details(prompt, **kwargs)
            return {**details, "usage": {"prompt_tokens": 11, "completion_tokens": 7}}

        def close(self):
            self.closed = True

    broken = json.dumps({
        "design": "constant score", "code": "def score(x):\n    raise ValueError('broken')\n",
    })
    audit = json.dumps({
        "relation": "new", "birth_id": None, "mechanism": "constant score",
        "evidence": "The first working scoring rule returns one.", "code_evidence": "return 1",
    })
    llm = ScriptedLLM("invalid JSON", broken, response(1), audit)
    monkeypatch.setattr(runner, "build_task", lambda *a, **kw: (TinyEvaluation(), {}))
    monkeypatch.setattr(runner, "build_llm_client", lambda **kw: llm)
    monkeypatch.setattr(runner, "run_in_tmux_log", lambda path, header, body: body())
    args = ["--task", "tsp_construct", "--budget", "3", "--roots", "1", "--run-name", "offline"]
    main(args)
    run_dir = tmp_path / "traceaad_v18_1" / "tsp_construct" / "offline"
    saved = read_json(run_dir / "run_config.json")
    summary = read_json(run_dir / "summary.json")
    assert saved["method"] == "traceaad_v18_1"
    assert saved["budget"] == saved["method_params"]["budget"] == 3
    assert saved["budget_axis"] == summary["budget_axis"] == "候选尝试"
    assert "BirthCheck" in saved["method_params"]["budget_basis"]
    assert saved["budget_basis"] == saved["method_params"]["budget_basis"]
    assert summary["status"] == "finished"
    assert summary["budget_used"] == 3
    assert summary["model_calls"] == 4
    assert summary["v18_1"]["birth_checks"] == 1
    assert summary["v18_1"]["auxiliary_calls_in_generation_budget"] is False
    assert summary["input_tokens"] == 44 and summary["output_tokens"] == 28
    calls = list(rows(run_dir / "calls.jsonl.gz"))
    assert [call["stage"] for call in calls] == ["init", "init", "repair", "birthcheck"]
    assert [call["sampling"]["max_tokens"] for call in calls] == [8192, 8192, 8192, 2048]
    assert llm.closed

    main([*args, "--dry-run"])
    preview = json.loads(capsys.readouterr().out)
    assert preview["budget_basis"] == saved["method_params"]["budget_basis"]
    assert preview["config"] == {
        key: value for key, value in saved["method_params"].items() if key != "budget_basis"
    }
    stored = (run_dir / "run_config.json").read_bytes()
    resumed_llm = ScriptedLLM()
    monkeypatch.setattr(runner, "build_llm_client", lambda **kw: resumed_llm)
    main(args)
    assert resumed_llm.closed and not resumed_llm.calls
    assert read_json(run_dir / "summary.json") == summary
    assert (run_dir / "run_config.json").read_bytes() == stored
