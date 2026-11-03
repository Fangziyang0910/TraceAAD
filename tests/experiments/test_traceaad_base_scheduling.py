from argparse import Namespace
import json
import os
import threading
import time

import pytest

from core.scheduling import SchedulerServer
from traceaad_base import Config, ModelReply, Task, TraceAAD
from experiments.traceaad_base.launch_host import build_plan
from experiments.traceaad_base.scheduling import Lease, ScheduledEvaluator, ScheduledModel, scheduler_status


@pytest.fixture
def pool(tmp_path, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    path = tmp_path / "scheduler.sock"
    cpu = min(os.sched_getaffinity(0))
    endpoint = {"base_url": "http://127.0.0.1:1/v1", "model": "test-model", "slots": 1}
    server = SchedulerServer(str(path), [cpu], [endpoint])
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
    thread.start()
    try:
        yield str(path), cpu
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_scheduler_reserves_before_budget_and_releases_both_resources(pool):
    path, cpu = pool
    seen = []

    class Client:
        def __init__(self, url, model, **options):
            assert options["request_options"]["chat_template_kwargs"] == {"enable_thinking": False}
            response_format = options["request_options"]["response_format"]
            assert response_format["type"] == "json_schema"
            assert response_format["json_schema"]["schema"]["required"] == ["analysis", "design", "code"]

        def complete(self, prompt, **options):
            seen.append(options)
            return ModelReply(json.dumps({"analysis": "Measure the CPU allocation.", "design": "Return the number of allowed CPUs",
                                           "code": "import os\ndef solve(x): return len(os.sched_getaffinity(0))"}))

    model = ScheduledModel(path, "base-test", client_factory=Client)
    task = Task("Minimize.", "def solve(x): pass", lambda s, f: f(0), 3)
    search = TraceAAD(task, model, config=Config(evaluation_budget=1))
    search.evaluator = ScheduledEvaluator(task, path, "base-test:search")
    affinity = os.sched_getaffinity(0)
    with Lease(path, "gpu", "blocker") as blocker:
        slot = blocker.acquire()
        errors = []
        def run():
            try:
                search.run()
            except BaseException as exc:
                errors.append(exc)
        thread = threading.Thread(target=run)
        thread.start()
        for _ in range(100):
            if scheduler_status(path)["gpu"]["waiting"]:
                break
            time.sleep(.01)
        assert search.model_calls == search.evaluations_used == 0
        blocker.poll(0, [slot])
        thread.join(timeout=10)
    assert not errors and not thread.is_alive()
    assert search.best.evaluation.fitness == 1
    assert search.evaluations_used == search.model_calls == 1
    assert seen == [{"temperature": 1., "max_tokens": 16384, "timeout": 600.}]
    assert search.attempts[0].calls[0]["metadata"]["scheduler"]["slot"] == "0:0"
    assert search.attempts[0].calls[0]["metadata"]["request_options"]["response_format"]["type"] == "json_schema"
    assert search.attempts[0].evaluations[0]["resources"]["cpu"] == cpu
    state = scheduler_status(path)
    assert state["cpu"]["used"] == state["gpu"]["used"] == 0
    assert os.sched_getaffinity(0) == affinity


def test_failed_model_request_is_counted_once_and_releases_slot(pool):
    path, _ = pool
    calls = []

    class FailingClient:
        def __init__(self, *args, **kwargs):
            pass

        def complete(self, *args, **kwargs):
            calls.append(1)
            raise TimeoutError("endpoint timeout")

    model = ScheduledModel(path, "failed-base", client_factory=FailingClient)
    task = Task("Minimize.", "def solve(x): pass", lambda s, f: f(0), 3)
    search = TraceAAD(task, model, config=Config(max_unevaluated=1))
    result = search.run()
    assert calls == [1] and result.model_calls == 1 and result.evaluations == 0
    record = search.attempts[0].calls[0]
    assert record["metadata"]["scheduler"]["endpoint"] == "http://127.0.0.1:1/v1"
    state = scheduler_status(path)
    assert state["gpu"]["used"] == 0 and state["gpu"]["endpoints"][0]["cooldown_seconds"] > 0


def test_eighteen_run_plan_uses_evaluation_budget_and_distinct_seeds():
    args = Namespace(batch="test_base_co6", experiment="traceaad_base", repeats=3,
                     budget=1000, initial_size=5, max_tokens=16384, model_timeout=600,
                     seconds_per_instance=10,
                     scheduler_socket="/tmp/scheduler.sock", env_file=None, skip_heldout=False)
    plan = build_plan(args)
    assert len(plan) == len({p["run_dir"] for p in plan}) == len({p["session"] for p in plan}) == 18
    assert len({p["task"] for p in plan}) == 6
    for task in {p["task"] for p in plan}:
        assert {p["seed"] for p in plan if p["task"] == task} == {0, 1, 2}
    for item in plan:
        command = item["command"]
        assert command[command.index("--budget") + 1] == "1000"
        assert command[command.index("--max-tokens") + 1] == "16384"
        assert command[command.index("--seconds-per-instance") + 1] == "10"
        assert "--skip-heldout" not in command


def test_run_enforces_explicit_training_deadline(pool, tmp_path, monkeypatch):
    from experiments.traceaad_base import run

    path, _ = pool

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def complete(self, *args, **kwargs):
            return ModelReply(json.dumps({
                "analysis": "Wait briefly and return zero.",
                "design": "Return zero after a delay.",
                "code": "import time\ndef solve(x):\n    time.sleep(1)\n    return 0",
            }))

    task = Task("Minimize.", "def solve(x): pass", lambda s, f: f(0), 5,
                metadata={"n_instance": 2, "settings": {"timeout_seconds": 5}})
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "load_task", lambda **kwargs: task)
    monkeypatch.setattr(run, "ScheduledModel", lambda p, label: ScheduledModel(p, label, client_factory=Client))
    run.main(["--task", "tsp_construct", "--run-name", "deadline_test", "--batch", "test",
              "--budget", "1", "--seconds-per-instance", ".025", "--skip-heldout",
              "--scheduler-socket", path])
    directory = tmp_path / "experiments_result/traceaad_base/tsp_construct/deadline_test"
    metadata = json.loads((directory / "run_config.json").read_text())
    evaluation = json.loads((directory / "native/evaluations/000001.json").read_text())
    state = json.loads((directory / "native/state.json").read_text())
    assert metadata["evaluation_execution"]["timeout_seconds"] == .05
    assert metadata["evaluation_execution"]["seconds_per_instance"] == .025
    assert metadata["task_eval"]["timeout_seconds"] == .05
    assert evaluation["evaluation"]["status"] == "timeout"
    assert state["evaluations_used"] == 1
    assert scheduler_status(path)["cpu"]["used"] == 0


def test_heldout_keeps_whole_task_timeout_and_frozen_program(pool, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from traceaad_base import Evaluation, Program
    from experiments.traceaad_base import run

    path, _ = pool
    external = SimpleNamespace(task_description="Minimize.", template_program="def solve(x): pass",
                               timeout_seconds=3, n_instance=2, evaluate_program=lambda source, f: f(3))
    monkeypatch.setattr(run, "heldout_task", lambda *a, **k: external)
    best = Program(7, "def solve(x): return x", "Identity", "Init", 0, None, Evaluation("success", 2))
    results = run.evaluate_heldout(tmp_path, "tsp_construct", best, path, evaluation_seed=11,
                                   seconds_per_instance=.5)
    assert len(results) == 1 and results[0]["fitness"] == 3
    assert results[0]["node_id"] == 7 and results[0]["timeout_scope"] == "whole_task"
    assert results[0]["memory_limit_bytes"] == 2 << 30
    assert results[0]["evaluation_seeds"] == [11]
    assert results[0]["timeout_seconds"] == 1
    assert results[0]["seconds_per_instance"] == .5
