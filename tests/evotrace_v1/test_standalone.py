"""EvoTrace must run with only its own package and the Python standard library."""

from dataclasses import asdict

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from evotrace_v1 import Task
from evotrace_v1.evaluation import Evaluator


def _copy_package(tmp_path):
    source = Path(__file__).resolve().parents[2] / "evotrace_v1"
    shutil.copytree(source, tmp_path / "evotrace_v1", ignore=shutil.ignore_patterns("__pycache__"))
    return {**os.environ, "PYTHONPATH": str(tmp_path)}


def test_package_alone_generates_repairs_reflects_and_resumes(tmp_path):
    env = _copy_package(tmp_path)
    script = r'''
from dataclasses import asdict
import importlib.abc
import json
from pathlib import Path
import sys

class NoRepositoryImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("traceaad", "core", "benchmarks", "experiments")):
            raise AssertionError("Repository dependency: " + fullname)

sys.meta_path.insert(0, NoRepositoryImports())
from evotrace_v1 import Config, EvoTrace, Task
from evotrace_v1.models import ModelReply
from evotrace_v1.topology import TopologyConfig

state = {"calls": 0, "reflections": 0}
class Model:
    request_options = {}
    def complete(self, prompt, **settings):
        schema = self.request_options["response_format"]["json_schema"]["name"]
        if schema == "evotrace_variation_experience":
            state["reflections"] += 1
            payload = {"experience": "Adjusted the offset. Earlier offsets had higher scores"}
        elif schema == "evotrace_selection_experience":
            state["reflections"] += 1
            payload = {"experience": "Earlier offsets had higher scores"}
        else:
            state["calls"] += 1
            n = state["calls"]
            if n == 1:
                return ModelReply("{invalid", usage={"total_tokens": 3})
            body = "raise ValueError('bad design')" if n == 3 else f"return {100-n}"
            payload = {"analysis": "Reduce the offset", "code": "def solve(x):\n    " + body + "\n", "design": "Return a constant offset"}
        return ModelReply(json.dumps(payload), usage={"total_tokens": 3})

task = Task("Minimize a finite score", "def solve(x):\n    pass\n", lambda src, fn: fn(1), 2)
config = Config(initial_size=3, evaluation_budget=12)
topology = TopologyConfig(base_capacity=4, trigger_capacity=7)
search = EvoTrace(task, Model(), config=config, topology_config=topology, run_dir="run")
for _ in range(8):
    assert search.step() is not None
assert search.consolidations_count and state["reflections"]
assert search.attempts[0].repair_used and search.attempts[1].repair_used
assert [e["status"] for e in search.attempts[1].evaluations] == ["error", "success"]
tree = asdict(search.tree)
for name in ("calls.jsonl", "attempts.jsonl"):
    with (Path("run") / name).open("ab") as f:
        f.write(b'{"partial":')
restored = EvoTrace(task, Model(), config=config, topology_config=topology, run_dir="run", resume=True)
assert asdict(restored.tree) == tree
assert restored.evaluations_used == search.evaluations_used
result = restored.run()
assert result.evaluations == 12 and result.best.valid
assert sum(restored.tree.direction_evaluations.values()) <= 12
for name in ("calls.jsonl", "attempts.jsonl"):
    for line in (Path("run") / name).read_text().splitlines():
        json.loads(line)
assert not any(name.startswith("traceaad") for name in sys.modules)
print(json.dumps({"evaluations": result.evaluations, "reflections": state["reflections"]}))
'''
    completed = subprocess.run([sys.executable, "-S", "-c", script], cwd=tmp_path, env=env,
                               text=True, capture_output=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout)["evaluations"] == 12


def test_independent_cli_uses_own_task_model_schema_and_resume(tmp_path):
    env = _copy_package(tmp_path)
    (tmp_path / "sample_task.py").write_text('''from evotrace_v1 import Task

def make_task():
    return Task("Minimize the score", "def solve(x):\\n    pass\\n", lambda src, fn: fn(1), 2)
''')
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            candidate = json.dumps({"analysis": "Use a constant", "code": "def solve(x):\n    return 1\n", "design": "Return one"})
            body = json.dumps({"choices": [{"message": {"content": candidate}, "finish_reason": "stop"}],
                               "usage": {"total_tokens": 11}}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        command = [sys.executable, "-S", "-m", "evotrace_v1", "--task-factory", "sample_task:make_task",
                   "--run-dir", "run", "--budget", "2", "--base-url", f"http://127.0.0.1:{server.server_port}/v1",
                   "--model", "local-test"]
        for extra in ([], ["--resume"]):
            completed = subprocess.run(command + extra, cwd=tmp_path, env=env,
                                       text=True, capture_output=True, timeout=30)
            assert completed.returncode == 0, completed.stdout + completed.stderr
            assert json.loads(completed.stdout)["evaluations"] == 2
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    assert len(requests) == 2  # Resume made no new model calls after budget exhaustion.
    assert requests[0]["response_format"]["json_schema"]["schema"]["required"] == ["analysis", "code", "design"]


@pytest.mark.parametrize("body,status", [
    ("return x + 1", "success"),
    ("return float('nan')", "error"),
    ("raise ValueError('invalid computation')", "error"),
    ("while True: pass", "timeout"),
])
def test_owned_evaluator_preserves_validity_and_time_limit(body, status):
    task = Task("Minimize the score", "def solve(x):\n    pass\n", lambda src, fn: fn(1), .2)
    result = Evaluator(task).evaluate("def solve(x):\n    " + body + "\n")
    assert result.status == status
    if status == "success":
        assert result.fitness == 2
    else:
        assert result.fitness is None and result.failure


def test_experiment_adapter_uses_evotrace_owned_runtime(tmp_path, monkeypatch, capsys):
    from experiments.traceaad import run
    from evotrace_v1 import OpenAIClient

    task = Task("Minimize the score", "def solve(x):\n    pass\n", lambda src, fn: fn(1), 2,
                metadata={"n_instance": 1, "settings": {"timeout_seconds": 2}})
    state = {"calls": 0}

    class Scheduled:
        model = "offline-test"

        def __init__(self, path, label):
            self.path, self.scheduler = path, {"endpoints": []}
            self.request_options = {}

        def complete(self, prompt, **settings):
            assert self.factory is OpenAIClient
            state["calls"] += 1
            return json.dumps({"analysis": "Return a finite score", "design": "Return a constant",
                               "code": f"def solve(x):\n    return {state['calls']}\n"})

    def forbidden_base_loader(**kwargs):
        raise AssertionError("EvoTrace must use its own task loader")

    def scheduled_evaluator(task, path, label, *, evaluator_factory):
        assert evaluator_factory is Evaluator
        return evaluator_factory(task)

    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "load_environment", lambda *_: None)
    monkeypatch.setattr(run, "load_task", forbidden_base_loader)
    monkeypatch.setattr(run, "load_evotrace_task", lambda **kwargs: task)
    monkeypatch.setattr(run, "ScheduledModel", Scheduled)
    monkeypatch.setattr(run, "ScheduledEvaluator", scheduled_evaluator)
    monkeypatch.setattr(run.signal, "signal", lambda *_: None)
    run.main(["--method", "evotrace.v1", "--task", "tsp_construct", "--run-name", "owned",
              "--batch", "owned", "--budget", "7", "--initial-size", "3",
              "--scheduler-socket", "/fake", "--skip-heldout"])
    result = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert result["evaluations"] == state["calls"] == 7
    directory = Path(result["run_dir"])
    saved = json.loads((directory / "native" / "run_config.json").read_text())
    assert "selection_temperature" not in saved["config"]
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["evaluation_calls"] == 7 and summary["status"] == "finished"
