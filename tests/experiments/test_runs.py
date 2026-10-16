"""Offline checks that experiment entries carry the actual evaluation settings."""

import importlib
import json
from dataclasses import asdict

import pytest

from baselines.shared.observability import shutdown_executor
from benchmarks.tasks import CLASSES, CO_TASKS, FUNCTION_LIMITS, INSTANCE_LIMITS, training_task
from experiments.infra import base, runner
from experiments.infra.search_run import build_parser
from tests.support import TinyEvaluation, TokenLLM, response
from traceaad import Config, TraceAAD


@pytest.mark.parametrize("task", CO_TASKS)
def test_training_and_test_share_the_task_budget(task, tmp_path):
    from benchmarks.tasks import PRIMARY_SPLITS
    from experiments.infra.evaluation_execution import heldout_evaluator

    train, _ = training_task(task, condition="traceaad")
    search = TraceAAD(evaluation=train, llm=TokenLLM(), run_dir=tmp_path, task=task)
    heldout, _ = heldout_evaluator(
        {
            "task": task,
            "method_params": {},
            "task_eval": {"interface": getattr(train, "interface", None)},
        },
        PRIMARY_SPLITS[task][0],
        (730241,),
        workers=1,
        timeout_seconds=None,
    )
    assert search.training.function_seconds == heldout.function_seconds == FUNCTION_LIMITS[task]
    assert search.training.timeout_seconds == heldout.timeout_seconds == INSTANCE_LIMITS[task]


@pytest.mark.parametrize(
    "name", ["eoh", "reevo", "mcts_ahd", "pathwise", "calm", "funsearch", "shinka_evo"]
)
def test_baseline_entry_accepts_and_evaluates_a_complete_solver(name, tmp_path, monkeypatch):
    monkeypatch.delenv("TRACEAAD_SCHEDULER_SOCKET", raising=False)
    log = tmp_path / "evaluations.jsonl"
    monkeypatch.setenv("TRACEAAD_EVALUATION_LOG", str(log))
    task = CLASSES["jssp_construct"](limit=1)
    module = importlib.import_module(f"experiments.{name}.run")
    monkeypatch.setattr(
        base, "training_task", lambda *a, **k: (task, {"interface": task.interface})
    )
    monkeypatch.setattr(module, "build_llm_client", lambda **k: TokenLLM())
    spec = module.make_run_spec(task="jssp_construct", eval_workers=1, experiments_root=tmp_path)
    method = module.build_method(spec, tmp_path / name / "logs")
    try:
        outcome = method._evaluator.evaluate_program_with_details(task.TEMPLATE)
        assert outcome.failure_kind is None, outcome.error
        assert outcome.result == task._rows[0]["reference"]
        record = json.loads(log.read_text())
        assert record["function_seconds_limit"] == FUNCTION_LIMITS["jssp_construct"]
        assert record["timeout_seconds"] == INSTANCE_LIMITS["jssp_construct"]
    finally:
        shutdown_executor(getattr(method, "_evaluation_executor", None))


@pytest.mark.parametrize("finished", [False, True])
def test_run_setup_updates_active_settings_and_preserves_completed_runs(
    tmp_path, monkeypatch, finished
):
    args = build_parser().parse_args(["--task", "tsp_construct", "--run-name", "same"])
    monkeypatch.setattr(runner, "build_task", lambda *a, **k: (TinyEvaluation(), {}))
    monkeypatch.setattr(runner, "build_llm_client", lambda **k: TokenLLM(response(1)))
    old = Config(budget=1 if finished else 4, roots=1, eval_timeout_seconds=1)
    ctx = runner.setup_experiment_run(
        args,
        method="traceaad",
        results_root=tmp_path,
        resume_file="events.jsonl",
        method_params=asdict(old),
    )
    method = TraceAAD(evaluation=ctx.evaluation, llm=ctx.llm, run_dir=ctx.run_dir, config=old)
    method.run() if finished else method._roots()
    before = (ctx.run_dir / "run_config.json").read_bytes()
    current = Config(budget=4, roots=1, eval_timeout_seconds=3, eval_workers=2)
    resumed = runner.setup_experiment_run(
        args,
        method="traceaad",
        results_root=tmp_path,
        resume_file="events.jsonl",
        method_params=asdict(current),
    )
    assert resumed.resumed
    if finished:
        assert (ctx.run_dir / "run_config.json").read_bytes() == before
    else:
        saved = json.loads((ctx.run_dir / "run_config.json").read_text())
        assert saved["method_params"]["eval_timeout_seconds"] == 3
        assert saved["method_params"]["eval_workers"] == 2


def test_traceaad_command_can_be_reviewed_without_calling_the_model(capsys):
    from experiments.traceaad.run import main

    main(["--task", "tsp_construct", "--budget", "7", "--eval-workers", "2", "--dry-run"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["config"]["budget"] == 7 and plan["config"]["eval_workers"] == 2
    assert plan["final_selection"] == "training"


def test_baseline_queue_skips_existing_runs_and_does_not_relaunch_finished_sessions(
    tmp_path, monkeypatch
):
    from experiments.infra import launcher, local_queue
    from traceaad.storage import write_json

    monkeypatch.setattr(local_queue, "RESULTS_ROOT", tmp_path)
    args = [
        "--method",
        "eoh",
        "--tasks",
        "tsp_construct",
        "op_aco",
        "cvrp_aco",
        "fssp_gls",
        "--repeats",
        "1",
        "--batch",
        "queue_test",
        "--slots",
        "1",
    ]
    plan = local_queue.build_plan(local_queue.build_parser().parse_args(args))
    write_json(plan[0].run_dir / "summary.json", {"status": "finished"})
    plan[1].run_dir.mkdir(parents=True)
    monkeypatch.setattr(launcher, "is_session_alive", lambda session: False)
    monkeypatch.setattr(local_queue, "is_session_alive", lambda session: False)
    monkeypatch.setattr(local_queue.time, "sleep", lambda seconds: None)
    launched = []
    monkeypatch.setattr(
        local_queue, "launch_command", lambda session, command: launched.append((session, command))
    )
    local_queue.main(args)
    assert [session for session, command in launched] == [p.session for p in plan[2:]]
    assert all(
        "experiments.eoh.run" in command and "--backend" in command for _, command in launched
    )
    assert not any(p.run_dir.exists() for p in plan[2:])
