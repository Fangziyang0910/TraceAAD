import json
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from traceaad_base import Config, Evaluator, OpenAIClient, Task, TraceAAD


@pytest.fixture
def endpoint():
    class Handler(BaseHTTPRequestHandler):
        mode = "success"
        requests = []

        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.requests.append((self.path, payload, self.headers.get("Authorization")))
            try:
                self.send_response(503 if self.mode == "failure" else 200)
                self.end_headers()
                if self.mode == "failure":
                    self.wfile.write(b"unavailable test-secret")
                elif self.mode == "slow":
                    for value in b'{"choices": []}':
                        self.wfile.write(bytes([value]))
                        self.wfile.flush()
                        time.sleep(.05)
                else:
                    self.wfile.write(json.dumps({"model": "test-model", "choices": [{
                        "message": {"content": json.dumps({"analysis": "Use a constant score.", "design": "Constant", "code": "def solve(x):\n    return 1"})},
                        "finish_reason": "stop"}], "usage": {"total_tokens": 41}}).encode())
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", Handler
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("request_options", [None, {"top_p": .8}])
def test_openai_request_records_settings_usage_and_does_not_persist_key(task, endpoint, tmp_path,
                                                                     request_options):
    url, handler = endpoint
    client = OpenAIClient(url, "test-model", api_key="test-secret", request_options=request_options)
    result = TraceAAD(task, client, config=Config(evaluation_budget=1), run_dir=tmp_path).run()
    assert result.best.evaluation.fitness == 1 and len(handler.requests) == 1
    path, payload, authorization = handler.requests[0]
    assert path == "/v1/chat/completions" and authorization == "Bearer test-secret"
    assert payload["temperature"] == 1 and payload["max_tokens"] == 16384
    assert payload["model"] == "test-model" and payload["stream"] is False
    response_format = payload["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["type"] == "object" and schema["additionalProperties"] is False
    assert list(schema["properties"]) == schema["required"] == ["analysis", "design", "code"]
    assert all(field == {"type": "string"} for field in schema["properties"].values())
    if request_options:
        assert payload["top_p"] == .8
    call = json.loads((tmp_path / "calls.jsonl").read_text().splitlines()[0])
    assert call["usage"]["total_tokens"] == 41 and call["finish_reason"] == "stop"
    assert all("test-secret" not in p.read_text() for p in tmp_path.rglob("*.json*"))


def test_request_failure_is_not_retried_and_error_body_redacts_key(task, endpoint, tmp_path):
    url, handler = endpoint
    handler.mode = "failure"
    client = OpenAIClient(url, "test-model", api_key="test-secret")
    result = TraceAAD(task, client, config=Config(max_unevaluated=1), run_dir=tmp_path).run()
    assert result.evaluations == 0 and result.model_calls == 1
    assert len(handler.requests) == 1
    call = json.loads((tmp_path / "calls.jsonl").read_text().splitlines()[0])
    assert "HTTP 503" in call["error"] and "[REDACTED]" in call["error"]
    assert all("test-secret" not in p.read_text() for p in tmp_path.rglob("*.json*"))


def test_model_deadline_limits_total_response_time(endpoint):
    url, handler = endpoint
    handler.mode = "slow"
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="total time limit"):
        OpenAIClient(url, "test-model").complete("test", temperature=1, max_tokens=16384, timeout=.2)
    assert time.monotonic() - started < .6
    assert len(handler.requests) == 1


def test_package_can_be_copied_renamed_and_run_without_repository_or_dependencies(tmp_path):
    source = Path(__file__).parents[2] / "traceaad_base"
    shutil.copytree(source.resolve(), tmp_path / "independent_aad",
                    ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "build"))
    script = textwrap.dedent('''
        import sys, importlib.abc, json
        sys.path.insert(0, sys.argv[1])
        class BlockRepository(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'traceaad', 'core', 'experiments', 'benchmarks', 'tests'}:
                    raise AssertionError('Repository dependency: ' + fullname)
        sys.meta_path.insert(0, BlockRepository())
        from independent_aad import Config, Task, TraceAAD
        class Model:
            def complete(self, prompt, **kwargs):
                return json.dumps({'analysis': 'Use the input directly.', 'design': 'Return the input', 'code': 'def solve(x): return x'})
        task = Task('Minimize.', 'def solve(x): pass', lambda source, f: f(2), 2)
        config = Config(evaluation_budget=4, initial_size=2)
        with TraceAAD(task, Model(), config=config, run_dir=sys.argv[2]) as first:
            first.step()
            first.step()
        resumed = TraceAAD(task, Model(), config=config, run_dir=sys.argv[2], resume=True)
        result = resumed.run()
        assert result.evaluations == result.model_calls == 4
        assert result.best.id == 1 and len(resumed.archive) == 4
        assert 'numpy' not in sys.modules
        print('independent search and resume passed')
    ''')
    result = subprocess.run([sys.executable, "-I", "-S", "-c", script, str(tmp_path), str(tmp_path / "run")],
                            cwd=tmp_path, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "independent search and resume passed" in result.stdout


@pytest.mark.parametrize("name", ["tsp_construct", "cvrp_aco", "op_aco", "online_bin_packing",
                                  "vrptw_construct", "fssp_gls", "graph_colouring", "jssp_construct"])
def test_existing_task_examples_execute_through_independent_evaluator(name):
    from benchmarks.tasks import CLASSES, FIXED_TASKS

    if name in FIXED_TASKS:
        options = {"settings": {"iterations": 1, "local_passes": 1}} if name == "fssp_gls" else {}
        external = CLASSES[name](limit=1, **options)
    elif name in {"cvrp_aco", "op_aco"}:
        external = CLASSES[name](n_ants=2, n_iterations=1, n_workers=1)
        external._datasets = external._datasets[:1]
        external.n_instance = 1
    elif name == "online_bin_packing":
        external = CLASSES[name](n_instances=1, n_items=20, capacity=100)
    else:
        external = CLASSES[name](n_instance=1, problem_size=10)
    task = Task.from_evaluator(external)
    result = Evaluator(task).evaluate(task.example_code)
    assert result.status == "success", result.failure
    namespace = {}
    exec(task.example_code, namespace)
    expected = external.evaluate_program(task.example_code, namespace[task.entrypoint])
    assert result.fitness == pytest.approx(expected)
    assert "pass" in task.function_template
