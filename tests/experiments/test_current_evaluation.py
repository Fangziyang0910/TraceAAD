"""Current execution across baseline entries, resume and frozen-program tests."""

import importlib
import json
import random
from dataclasses import asdict

import pytest

from core import Function, SecureEvaluator
from core.evaluate import EVALUATION_SEED
from core.scheduling import SchedulerError
from baselines.observability import is_search_aborted
from baselines.profiler import ProfilerBase
from benchmarks.tasks import FUNCTION_SECONDS, INSTANCE_SECONDS
from experiments.infra import base, evaluate, runner, search_heldout
from tests.method.test_instance_evaluation import IndexedEvaluation
from tests.support import TokenLLM, response
from traceaad.common.config import REVISION
from traceaad.v10_21 import Config, TraceAADV1021


@pytest.mark.parametrize("name", ["eoh", "reevo", "mcts_ahd", "pathwise", "calm", "funsearch", "shinka_evo"])
def test_baseline_entries_use_instance_seeds_globals_and_deadlines(name, tmp_path, monkeypatch):
    monkeypatch.delenv("TRACEAAD_SCHEDULER_SOCKET", raising=False)
    monkeypatch.setenv("TRACEAAD_EVALUATION_LOG", str(tmp_path / "evaluations.jsonl"))
    module = importlib.import_module(f"experiments.{name}.run")
    task = IndexedEvaluation(2)
    monkeypatch.setattr(base, "training_task", lambda *args, **kwargs: (task, {}))
    monkeypatch.setattr(module, "build_llm_client", lambda **kwargs: TokenLLM())
    spec = module.make_run_spec(task="tsp_construct", eval_workers=2, experiments_root=tmp_path)
    method = module.build_method(spec, tmp_path / name / "logs")
    code = """import random
count = 0
def score(x):
    global count
    count += 1
    assert count == 1
    return random.random()
"""
    result = method._evaluator.evaluate_program_with_details(code)
    expected = sum(random.Random(EVALUATION_SEED + i).random() for i in range(2)) / 2
    assert result.failure_kind is None and result.result == expected
    task._instance_execution["timeout_seconds"] = .1
    result = method._evaluator.evaluate_program_with_details("import time\ndef score(x):\n    time.sleep(.3)\n    return 1")
    assert result.result is None and result.failure_kind == "timeout"
    rows = [json.loads(line) for line in (tmp_path / "evaluations.jsonl").read_text().splitlines()]
    assert [(row["valid"], row["failure_kind"], row["error_type"]) for row in rows] == [
        (True, None, None), (False, "timeout", "TimeoutError")]
    assert rows[0]["function_seconds_limit"] == FUNCTION_SECONDS
    metadata = runner.baseline_run_config(spec, tmp_path / name, "test", name,
                                           {"max_sample_nums": 1}, task_config={})
    assert metadata["method_params"]["evaluation_seeds"] == [EVALUATION_SEED]
    assert metadata["method_params"]["eval_timeout_seconds"] == INSTANCE_SECONDS
    assert metadata["method_params"]["function_seconds"] == FUNCTION_SECONDS
    assert metadata["method_params"]["eval_workers"] == 2
    task._instance_execution["scheduler_socket"] = str(tmp_path / "missing.socket")
    with pytest.raises(SchedulerError):
        method._evaluator.evaluate_program_with_details("def score(x): return x")
    assert is_search_aborted(method)
    continues = getattr(method, "_has_budget", None) or method._continue_loop
    assert not continues()
    method._profiler.write_run_summary(status="finished")
    summary = json.loads((method._profiler.run_dir / "summary.json").read_text())
    assert summary["status"] == "aborted" and summary["error_type"] == "SchedulerError"


@pytest.mark.parametrize("finished", [False, True])
def test_resume_updates_active_parameters_and_keeps_finished_config(finished, tmp_path, monkeypatch):
    from experiments.infra.search_run import build_parser
    args = build_parser("test", Config).parse_args(["--task", "tsp_construct", "--run-name", "same"])
    monkeypatch.setattr(runner, "build_task", lambda *args, **kwargs: (IndexedEvaluation(2), {}))
    monkeypatch.setattr(runner, "build_llm_client", lambda **kwargs: TokenLLM(response(1)))
    old = Config(budget=1 if finished else 4, roots=1, eval_timeout_seconds=1)
    ctx = runner.setup_experiment_run(args, method="v1021", results_root=tmp_path, resume_file="events.jsonl",
                                      method_params=asdict(old))
    method = TraceAADV1021(evaluation=ctx.evaluation, llm=ctx.llm, run_dir=ctx.run_dir, config=old)
    if finished:
        method.run()
    else:
        method._roots()
    before = (ctx.run_dir / "run_config.json").read_bytes()
    current = Config(budget=4, roots=1, eval_timeout_seconds=3, eval_workers=2)
    resumed = runner.setup_experiment_run(args, method="v1021", results_root=tmp_path, resume_file="events.jsonl",
                                          method_params=asdict(current))
    assert resumed.resumed
    if finished:
        assert (ctx.run_dir / "run_config.json").read_bytes() == before
    else:
        saved = json.loads((ctx.run_dir / "run_config.json").read_text())
        assert saved["method_params"]["eval_timeout_seconds"] == 3
        assert saved["method_params"]["eval_workers"] == 2


@pytest.mark.parametrize("entry", ["search", "generic"])
def test_existing_heldout_scores_are_returned_without_protocol_checks(entry, tmp_path, monkeypatch):
    method = TraceAADV1021(evaluation=IndexedEvaluation(2), llm=TokenLLM(response(1)), run_dir=tmp_path,
                           config=Config(budget=1, roots=1))
    summary = method.run()
    (tmp_path / "run_config.json").write_text(json.dumps({"task": "tsp_construct", "method_params": {}}))
    variant = "" if entry == "search" else f"shared:{REVISION}"
    previous = {"task": "tsp_construct", "split": "eval", "scale": "50", "variant": variant,
                "key": summary["best"]["key"], "node_id": 1, "verification": "verified",
                "fitness": 432.1, "protocol": "different", "timeout_seconds": 987}
    path = tmp_path / "heldout.json"
    path.write_text(json.dumps([previous]))
    before = path.read_bytes()

    def should_not_evaluate(*args, **kwargs):
        raise AssertionError("existing results must not be reevaluated")

    module = search_heldout if entry == "search" else evaluate
    monkeypatch.setattr(module, "heldout_task", should_not_evaluate)
    result = (module.evaluate_run(tmp_path, timeout_seconds=.1) if entry == "search" else
              module.evaluate_run(tmp_path, ["eval"], timeout_seconds=.1)[0])
    assert result == previous and path.read_bytes() == before


def test_new_baseline_heldout_uses_the_current_executor(tmp_path, monkeypatch):
    method = TraceAADV1021(evaluation=IndexedEvaluation(2), llm=TokenLLM(response(1)), run_dir=tmp_path,
                           config=Config(budget=1, roots=1))
    method.run()
    (tmp_path / "run_config.json").write_text(json.dumps({"task": "tsp_construct", "method": "eoh",
                                                         "method_params": {}}))
    monkeypatch.setattr(evaluate, "heldout_task", lambda *args: IndexedEvaluation(2))
    result = evaluate.evaluate_run(tmp_path, ["eval"])[0]
    assert result["timeout_scope"] == "instance" and result["timeout_seconds"] == INSTANCE_SECONDS
    assert result["function_seconds"] == FUNCTION_SECONDS
    assert result["evaluation_seeds"] == [EVALUATION_SEED] and result["fitness"] == 1


def test_baseline_freeze_rechecks_five_distinct_programs_and_preserves_candidates(tmp_path):
    from types import SimpleNamespace
    from traceaad.common.storage import selected_program
    from experiments.infra.monitor_history import TrainingHistory
    task = IndexedEvaluation(1)
    task._instance_execution = {"timeout_seconds": 5, "n_workers": 1}
    profiler = ProfilerBase(tmp_path)
    profiler.record_parameters(None, task, SimpleNamespace(_evaluator=SecureEvaluator(task)))
    (tmp_path / "run_config.json").write_text(json.dumps({"task": "tsp_construct", "method": "eoh"}))
    for value, score in [(9, -100), (8, -90), (7, -80), (6, -70), (5, -60), (4, -50), (9, -110)]:
        function = Function(name="score", args="x", body=f"    return {value}")
        function.score, function.operator = score, "test"
        profiler.register_function(function, program=f"def score(x):\n    return {value}\n")
    before = (tmp_path / "events.jsonl").read_bytes()
    profiler.finish()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "finished" and summary["best"]["fitness"] == 5
    assert summary["best"]["id"] == 5 and summary["selection_evaluations"] == 5
    assert summary["candidate_count"] == summary["budget_used"] == 7
    assert summary["evaluation_calls"] == 12
    assert (tmp_path / "events.jsonl").read_bytes().startswith(before)
    assert selected_program(tmp_path)["verified"]
    history = TrainingHistory(tmp_path, minimize=True)
    history.read()
    assert history.progress_snapshot()["candidate_count"] == 7


@pytest.mark.parametrize("candidates", [0, 2])
def test_baseline_freeze_without_a_valid_result_is_not_finished(candidates, tmp_path):
    from types import SimpleNamespace
    from experiments.infra.monitor_history import TrainingHistory
    task = IndexedEvaluation(1)
    task._instance_execution = {"timeout_seconds": 5, "n_workers": 1}
    profiler = ProfilerBase(tmp_path)
    profiler.record_parameters(None, task, SimpleNamespace())
    for index in range(candidates):
        body = f"    raise ValueError('bad{index}')"
        function = Function(name="score", args="x", body=body)
        function.score, function.operator = index, "test"
        profiler.register_function(function, program=f"def score(x):\n{body}\n")
    profiler.finish()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == ("selection_failed" if candidates else "no_valid_root")
    assert summary["best"] is None
    assert not (tmp_path / "best_program.py").exists()
    TrainingHistory(tmp_path, minimize=True).read()


@pytest.mark.parametrize("first", ["generic", "search"])
def test_heldout_entries_share_one_result(first, tmp_path, monkeypatch):
    method = TraceAADV1021(evaluation=IndexedEvaluation(1), llm=TokenLLM(response(1)), run_dir=tmp_path,
                           config=Config(budget=1, roots=1))
    method.run()
    (tmp_path / "run_config.json").write_text(json.dumps({"task": "tsp_construct", "method_params": {}}))
    for module in (search_heldout, evaluate):
        monkeypatch.setattr(module, "heldout_task", lambda *args: IndexedEvaluation(1))
    read = {"generic": lambda: evaluate.evaluate_run(tmp_path, ["eval_50"])[0],
            "search": lambda: search_heldout.evaluate_run(tmp_path)}
    result = read[first]()
    path = tmp_path / "heldout.json"
    before = path.read_bytes()

    def should_not_evaluate(*args, **kwargs):
        raise AssertionError("the other entry must read the existing result")

    for module in (search_heldout, evaluate):
        monkeypatch.setattr(module, "heldout_task", should_not_evaluate)
    second = "generic" if first == "search" else "search"
    assert read[second]() == result and path.read_bytes() == before
    assert len(json.loads(path.read_text())) == 1


def test_local_queue_advances_when_a_run_exits_before_writing_results(tmp_path, monkeypatch):
    from experiments.infra import local_queue
    plan = [base.LaunchItem(task="tsp_construct", repeat=1, backend="local", session=name,
                           run_name=name, run_dir=tmp_path / name, seed=0, module="experiments.funsearch.run")
            for name in ("first", "second")]
    launched = []
    monkeypatch.setattr(local_queue, "build_plan", lambda args: plan)
    monkeypatch.setattr(local_queue, "is_session_alive", lambda session: False)
    monkeypatch.setattr(base, "item_is_running", lambda item: False)
    monkeypatch.setattr(local_queue.time, "sleep", lambda seconds: None)

    def launch(session, command):
        assert session not in launched, "an exited run must not block the rest of the queue"
        launched.append(session)

    monkeypatch.setattr(local_queue, "launch_command", launch)
    local_queue.main(["--method", "funsearch", "--batch", "test", "--slots", "1"])
    assert launched == ["first", "second"]
