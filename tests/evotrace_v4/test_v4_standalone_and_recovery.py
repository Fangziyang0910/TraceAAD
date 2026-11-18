"""EvoTrace v4 standalone stdlib isolation, budget gatekeeping, init readiness, and deterministic resume."""

from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

import pytest

from evotrace_v4 import Config, EvoTrace, OpenAIClient, Task
from evotrace_v4.evaluation import Evaluator
from evotrace_v4.topology import TopologyConfig


def _copy_package(tmp_path):
    source = Path(__file__).resolve().parents[2] / "evotrace_v4"
    shutil.copytree(source, tmp_path / "evotrace_v4", ignore=shutil.ignore_patterns("__pycache__"))
    return {**os.environ, "PYTHONPATH": str(tmp_path)}


class ScriptedModel:
    request_options = {}

    def __init__(self, fail_on_calls=()):
        self.state = {"calls": 0, "reflections": 0}
        self.fail_on = set(fail_on_calls)

    def complete(self, prompt, **kwargs):
        schema = self.request_options["response_format"]["json_schema"]["name"]
        if schema == "evotrace_experience":
            self.state["reflections"] += 1
            return json.dumps({"experience": "Summarized factual computations and outcomes."})
        self.state["calls"] += 1
        n = self.state["calls"]
        if n in self.fail_on:
            body = "raise ValueError('bad computation')"
        else:
            body = f"return (x + {n}) if x >= 0 else {n}"
        return json.dumps({"design": f"Offset computation {n}", "code": f"def solve(x):\n    {body}\n"})


def test_last_evaluation_never_overshoots_budget_on_failed_candidate(tmp_path):
    """复核末步预算守门：最后只剩 1 次评价且该候选运行失败时，绝不发起修复重评导致超支。"""
    task = Task("Minimize", "def solve(x):\n    pass\n", lambda src, fn: fn(1), 2)
    # Budget is 4; call 4 is the 4th evaluation and fails execution -> must stop at exactly 4 evaluations!
    model = ScriptedModel(fail_on_calls={4})
    search = EvoTrace(task, model, config=Config(initial_size=3, evaluation_budget=4, seed=1), run_dir=tmp_path / "b4")
    result = search.run()
    assert result.evaluations == 4
    assert model.state["calls"] == 4  # No wasted repair LLM call when budget is already 4/4!
    assert not search.attempts[-1].repair_used


def test_initialization_completion_computes_core_set_and_simplifies_excess_init_failures(tmp_path):
    """复核首步就绪：集齐 N_0 个有效初始个体时立即计算 S 并对满块初始失败旁支执行根部维护。"""
    task = Task("Minimize", "def solve(x):\n    pass\n", lambda src, fn: fn(1), 2)
    # Calls 1..8 are 4 failed init attempts (each repaired once and still failing -> 4 failed root nodes, 8 evals),
    # then calls 9, 10 are 2 valid init attempts -> initial_size=2 reached!
    model = ScriptedModel(fail_on_calls={1, 2, 3, 4, 5, 6, 7, 8})
    search = EvoTrace(
        task,
        model,
        config=Config(initial_size=2, evaluation_budget=20, seed=0),
        topology_config=TopologyConfig(max_representatives=8, cover_distance=0.0, region_size=4),
        run_dir=tmp_path / "init_clean",
    )
    for _ in range(6):
        search.step()
    assert search._initialized_count == 2
    assert len(search.core_ids) == 2
    assert search.tree.root_experience == "Summarized factual computations and outcomes."
    assert len(search.tree.nodes) == 2  # The 4 failed init nodes were pruned on init completion!


def test_all_evolution_operators_attach_to_living_parents_and_resume_identically(tmp_path):
    task = Task("Minimize", "def solve(x):\n    pass\n", lambda src, fn: fn(1), 2)
    config = Config(initial_size=3, evaluation_budget=60, seed=7)
    topology = TopologyConfig(max_representatives=8, cover_distance=0.15, region_size=3)

    full = EvoTrace(task, ScriptedModel(), config=config, topology_config=topology, run_dir=tmp_path / "full")
    attached = []
    orig_attach = full._attach_evaluated_attempt

    def track(attempt):
        orig_attach(attempt)
        if attempt.program_id in full.tree.edges:
            attached.append((attempt.operator, attempt.parent_id, full.tree.edges[attempt.program_id].parent_id))

    full._attach_evaluated_attempt = track
    full.run()
    ops = {op for op, _, _ in attached}
    assert {"Init", "Refine", "Explore"} <= ops
    for op, parent_id, edge_parent in attached:
        assert parent_id == edge_parent
        if op == "Init":
            assert parent_id == 0
        else:
            assert parent_id > 0

    # Split at 30 steps, corrupt trailing bytes, and verify exact deterministic resume to 60 evaluations!
    part_model = ScriptedModel()
    part = EvoTrace(task, part_model, config=config, topology_config=topology, run_dir=tmp_path / "split")
    for _ in range(30):
        assert part.step() is not None
    for name in ("calls.jsonl", "attempts.jsonl"):
        with (tmp_path / "split" / name).open("ab") as f:
            f.write(b'{"truncated":')
    calls_before = dict(part_model.state)
    resumed = EvoTrace(task, part_model, config=config, topology_config=topology, run_dir=tmp_path / "split", resume=True)
    assert resumed.tree == part.tree
    assert part_model.state == calls_before
    resumed.run()
    assert resumed.tree == full.tree
    assert resumed.evaluations_used == full.evaluations_used == 60


def test_package_alone_runs_with_python_stdlib_only(tmp_path):
    env = _copy_package(tmp_path)
    script = r'''
import importlib.abc
import json
import sys

class NoRepositoryImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("traceaad", "evotrace_v1", "evotrace_v2", "evotrace_v3", "core", "benchmarks", "experiments")):
            raise AssertionError("Repository dependency: " + fullname)

sys.meta_path.insert(0, NoRepositoryImports())
from evotrace_v4 import Config, EvoTrace, Task
from evotrace_v4.models import ModelReply
from evotrace_v4.topology import TopologyConfig

state = {"calls": 0, "reflections": 0}
class Model:
    request_options = {}
    def complete(self, prompt, **settings):
        schema = self.request_options["response_format"]["json_schema"]["name"]
        if schema == "evotrace_experience":
            state["reflections"] += 1
            return ModelReply(json.dumps({"experience": "Observed offset effects."}), usage={"total_tokens": 3})
        state["calls"] += 1
        n = state["calls"]
        if n == 1:
            return ModelReply("{broken", usage={"total_tokens": 2})
        body = "raise ValueError('bad')" if n == 3 else f"return {50 - n}"
        return ModelReply(json.dumps({"design": f"Design {n}", "code": f"def solve(x):\n    {body}\n"}), usage={"total_tokens": 4})

task = Task("Minimize", "def solve(x):\n    pass\n", lambda src, fn: fn(1), 2)
search = EvoTrace(task, Model(), config=Config(initial_size=2, evaluation_budget=10),
                  topology_config=TopologyConfig(max_representatives=6, cover_distance=0.15, region_size=2),
                  run_dir="run")
result = search.run()
assert result.evaluations == 10 and result.best.valid and state["reflections"] >= 1
print(json.dumps({"evaluations": result.evaluations, "reflections": state["reflections"]}))
'''
    completed = subprocess.run([sys.executable, "-S", "-c", script], cwd=tmp_path, env=env,
                               text=True, capture_output=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout)["evaluations"] == 10


def test_independent_cli_and_experiment_adapter_end_to_end(tmp_path, monkeypatch, capsys):
    env = _copy_package(tmp_path)
    (tmp_path / "sample_task.py").write_text('''from evotrace_v4 import Task
def make_task():
    return Task("Minimize", "def solve(x):\\n    pass\\n", lambda src, fn: fn(1), 2)
''')
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            candidate = json.dumps({"design": "Return one", "code": "def solve(x):\n    return 1\n"})
            body = json.dumps({"choices": [{"message": {"content": candidate}, "finish_reason": "stop"}],
                               "usage": {"total_tokens": 9}}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        command = [sys.executable, "-S", "-m", "evotrace_v4", "--task-factory", "sample_task:make_task",
                   "--run-dir", "cli_run", "--budget", "2", "--base-url", f"http://127.0.0.1:{server.server_port}/v1",
                   "--model", "local-test"]
        for extra in ([], ["--resume"]):
            completed = subprocess.run(command + extra, cwd=tmp_path, env=env, text=True, capture_output=True, timeout=30)
            assert completed.returncode == 0, completed.stdout + completed.stderr
            assert json.loads(completed.stdout)["evaluations"] == 2
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    assert len(requests) == 2
    assert requests[0]["response_format"]["json_schema"]["schema"]["required"] == ["design", "code"]

    # Also test unified experiment runner with evotrace.v4!
    from experiments.traceaad import run

    task = Task("Minimize", "def solve(x):\n    pass\n", lambda src, fn: fn(1), 2,
                metadata={"n_instance": 1, "settings": {"timeout_seconds": 2}})
    state = {"calls": 0, "reflections": 0}

    class Scheduled:
        model = "offline-test"

        def __init__(self, path, label):
            self.path, self.scheduler = path, {"endpoints": []}
            self.request_options = {}

        def complete(self, prompt, **settings):
            assert self.factory is OpenAIClient
            schema = self.request_options.get("response_format", {}).get("json_schema", {}).get("name", "")
            if schema == "evotrace_experience":
                state["reflections"] += 1
                return json.dumps({"experience": "Condensed attempt outcomes."})
            state["calls"] += 1
            return json.dumps({"design": f"Constant {state['calls']}", "code": f"def solve(x):\n    return {state['calls']}\n"})

    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "load_environment", lambda *_: None)
    monkeypatch.setattr(run, "load_evotrace_v4_task", lambda **kwargs: task)
    monkeypatch.setattr(run, "ScheduledModel", Scheduled)
    monkeypatch.setattr(run, "ScheduledEvaluator", lambda t, p, l, *, evaluator_factory: evaluator_factory(t))
    monkeypatch.setattr(run.signal, "signal", lambda *_: None)

    run.main(["--method", "evotrace.v4", "--task", "tsp_construct", "--run-name", "v4_run",
              "--batch", "v4_batch", "--budget", "8", "--initial-size", "3",
              "--scheduler-socket", "/fake", "--skip-heldout", "--region-size", "2"])
    out = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert out["evaluations"] == 8 and state["reflections"] >= 1
    summary = json.loads((Path(out["run_dir"]) / "summary.json").read_text())
    assert summary["method"] == "evotrace_v4" and summary["status"] == "finished"
