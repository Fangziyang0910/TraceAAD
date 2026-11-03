import importlib.abc
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

import pytest

from traceaad_v19_3 import Config, OpenAIClient, TraceAAD
from traceaad_v19_3.prompts import ContextOverflow, PromptBuilder


EMPTY = '{"positive": [], "negative": []}'


def test_actual_http_request_uses_role_specific_schema_and_limits(task, counter, tmp_path, monkeypatch):
    script = r'''
from http.server import BaseHTTPRequestHandler, HTTPServer
import json,sys
from pathlib import Path
posted=[]
class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        posted.append(request)
        Path(sys.argv[1]).write_text(json.dumps(posted))
        reflection=request['response_format']['json_schema']['name'].endswith('reflector')
        content='{"positive":[],"negative":[]}' if reflection else json.dumps({
            'code':f'def solve(x): return {6-len(posted)}',
            'analysis':'Reduce the constant.','design':'Use a smaller constant.'})
        data=json.dumps({'choices':[{'message':{'content':content},'finish_reason':'stop'}],
            'model':'test-http','usage':{'prompt_tokens':11,'completion_tokens':7}}).encode()
        self.send_response(200)
        self.send_header('Content-Length',str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def log_message(self,*args):pass
server=HTTPServer(('127.0.0.1',0),Handler)
print(server.server_port,flush=True)
server.serve_forever()
'''
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    server = subprocess.Popen([sys.executable, "-u", "-c", script, str(tmp_path / "posted.json")],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        port = int(server.stdout.readline())
        model = OpenAIClient(f"http://127.0.0.1:{port}/v1", "test-http",
                            request_options={"response_format": {"type": "text"}})
        with TraceAAD(task, model, config=Config(evaluation_budget=2, initial_size=1),
                      token_counter=counter, run_dir=tmp_path) as search:
            result = search.run()
            assert result.model_calls == 3
            assert search.attempts[1].calls[-1]["usage"]["completion_tokens"] == 7
        posted = json.loads((tmp_path / "posted.json").read_text())
        assert [r["max_tokens"] for r in posted] == [16384, 16384, 2048]
        assert posted[0]["response_format"]["json_schema"]["schema"]["required"] == ["code", "analysis", "design"]
        assert posted[-1]["response_format"]["json_schema"]["schema"]["required"] == ["positive", "negative"]
    finally:
        server.terminate()
        server.communicate(timeout=5)


def test_detached_package_search_and_resume_without_other_repo_modules(tmp_path):
    root = Path(__file__).resolve().parents[2]
    shutil.copytree(root / "traceaad_v19_3", tmp_path / "detached", ignore=shutil.ignore_patterns("__pycache__"))
    script = r'''
import importlib.abc, json, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'traceaad','traceaad_base','traceaad_v19_2','traceaad_v19_3','core','benchmarks','experiments'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from detached import Config, Task, TraceAAD
class Count:
    identity = {'test': 'words'}
    def __call__(self, s): return len(s.split())
class Model:
    def complete(self, prompt, **settings):
        if settings['response_format']['json_schema']['name'].endswith('reflector'):
            return '{"positive": [], "negative": []}'
        return json.dumps({'code':'def solve(x): return 1', 'analysis':'Use a constant.', 'design':'A constant rule.'})
task=Task('Minimize the numeric result.', 'def solve(x): return x', lambda code, fn: fn(1), time_limit=3)
with TraceAAD(task,Model(),config=Config(evaluation_budget=2,initial_size=1),run_dir='run',token_counter=Count()) as s:
    s.step()
with TraceAAD(task,Model(),run_dir='run',resume=True,token_counter=Count()) as s:
    r=s.run()
    assert (r.evaluations,r.model_calls,r.valid_programs)==(2,3,1)
'''
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)}, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_initialization_is_sequential_and_distinct(task, model, reply, counter):
    llm = model([reply(5), reply(5), reply(3)])
    with TraceAAD(task, llm, config=Config(evaluation_budget=3, initial_size=2), token_counter=counter) as search:
        result = search.run()
        assert result.valid_programs == 2
        assert [a.operator for a in search.attempts] == ["Init"] * 3
        assert search.attempts[0].context["initial_context_ids"] == []
        assert search.attempts[1].context["initial_context_ids"] == [1]
        assert search.attempts[2].context["initial_context_ids"] == [1]


def test_format_repair_checks_whole_request_capacity(task, counter):
    builder = PromptBuilder(task, Config(context_window=300, max_tokens=32,
        reflection_max_tokens=32, protocol_tokens=8), counter)
    with pytest.raises(ContextOverflow):
        builder.repair_format("broken " * 1000, "Expected JSON")


def test_experiment_previews_are_read_only_and_support_reflection_modes(capsys):
    from experiments.traceaad_v19_3.run import main as run
    from experiments.traceaad_v19_3.launch_host import main as launch
    run(["--task", "tsp_construct", "--run-name", "preview_v193", "--batch", "preview_v193",
         "--budget", "3", "--reflection-mode", "local", "--dry-run"])
    metadata = json.loads(capsys.readouterr().out)
    assert metadata["method"] == "traceaad_v19_3"
    assert metadata["method_params"]["reflection_mode"] == "local"
    assert metadata["evaluation_execution"]["timeout_scope"] == "whole_task"
    assert metadata["evaluation_execution"]["timeout_seconds"] == 160
    launch(["--batch", "preview_v193", "--tasks", "tsp_construct", "--repeats", "1",
            "--reflection-mode", "siblings"])
    preview = json.loads(capsys.readouterr().out)
    assert len(preview["plan"]) == 1 and preview["status"] == "planned"
    assert "siblings" in preview["plan"][0]["command"]
    assert not Path(preview["manifest"]).exists()


def test_reporter_separates_generation_and_reflection_usage(task, model, reply, counter, tmp_path):
    from experiments.traceaad_v19_3.reporting import Reporter
    llm = model([reply(5), reply(3), EMPTY])
    with TraceAAD(task, llm, config=Config(evaluation_budget=2, initial_size=1),
                  token_counter=counter, run_dir=tmp_path / "native") as search:
        result = search.run()
        summary = Reporter(tmp_path, {"method": "traceaad_v19_3"}).publish(search, result)
        assert summary["model_calls_by_role"] == {"generator": 2, "reflector": 1}
        assert summary["evaluation_calls"] == 2
        assert summary["token_usage"]["calls_without_usage"] == 3
        assert (tmp_path / "best_program.py").read_text() == result.best.code
        assert json.loads((tmp_path / "selection.json").read_text())["selected_node"] == result.best.id


def test_task_source_is_snapshotted_and_incompatible_resume_is_rejected(task, model, counter, tmp_path):
    source = tmp_path / "evaluation_source.py"
    source.write_text("PARAMETER = 1\n")
    task.metadata["source_hashes"] = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
    run = tmp_path / "run"
    with TraceAAD(task, model([]), config=Config(evaluation_budget=1), token_counter=counter, run_dir=run):
        pass
    manifest = json.loads((run / "source/task_sources.json").read_text())
    assert len(manifest) == 1
    assert (run / "source" / manifest[0]["snapshot"]).read_text() == "PARAMETER = 1\n"
    source.write_text("PARAMETER = 2\n")
    with pytest.raises(ValueError, match="matching"):
        TraceAAD(task, model([]), token_counter=counter, run_dir=run, resume=True)


def test_scheduled_model_passes_both_schemas_and_releases_each_lease(task, counter, monkeypatch):
    from traceaad_v19_3 import ModelReply
    from experiments.traceaad_v19_3 import scheduling
    status = {"instance_id": "test", "endpoints": [{"base_url": "http://127.0.0.1/v1", "model": "test"}]}
    releases, received = [], []
    class Lease:
        def __init__(self, *args):pass
        def __enter__(self):return self
        def __exit__(self, *args):pass
        def status(self):return status
        def acquire(self):return "0:slot"
        def poll(self, target, release=(), **kwargs):
            releases.extend(release)
            return []
    class Client:
        def __init__(self, *args, **kwargs):pass
        def complete(self, prompt, **settings):
            received.append(settings)
            reflection = settings["response_format"]["json_schema"]["name"].endswith("reflector")
            content = EMPTY if reflection else json.dumps({"code": f"def solve(x): return {6-len(received)}",
                "analysis": "Reduce the constant.", "design": "Use a smaller constant."})
            return ModelReply(content)
    monkeypatch.setattr(scheduling, "scheduler_status", lambda _: status)
    monkeypatch.setattr(scheduling, "Lease", Lease)
    model = scheduling.ScheduledModel("unused", "test", client_factory=Client)
    with TraceAAD(task, model, config=Config(evaluation_budget=2, initial_size=1), token_counter=counter) as search:
        result = search.run()
        assert result.model_calls == 3
        assert search.attempts[1].calls[-1]["metadata"]["request_options"]["response_format"] == received[-1]["response_format"]
    assert releases == ["0:slot"] * 3
    assert received[-1]["response_format"]["json_schema"]["schema"]["required"] == ["positive", "negative"]


@pytest.mark.parametrize("heldout_failure", [False, True])
def test_experiment_runner_freezes_training_winner_before_heldout(task, model, reply, reflection, tmp_path, monkeypatch, capsys, heldout_failure):
    from experiments.traceaad_v19_3 import run
    task.metadata.update(n_instance=1, settings={})
    llm = model([reply(5), reply(3), reflection, reply(2), EMPTY])
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "load_task", lambda **kwargs: task)
    monkeypatch.setattr(run, "OpenAIClient", lambda *args, **kwargs: llm)
    tested = []
    def heldout(directory, name, best, scheduler, **kwargs):
        state = json.loads((directory / "native/state.json").read_text())
        selection = json.loads((directory / "selection.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        assert all(a["completed"] for a in state["attempts"])
        assert selection["selected_node"] == best.id
        assert summary["status"] == "running" and summary["phase"] == "heldout"
        assert "finished_at" not in summary and summary["training_finished_at"]
        tested.append(best)
        if heldout_failure:
            raise RuntimeError("held-out evaluation failed")
    monkeypatch.setattr(run, "evaluate_heldout", heldout)
    previous = signal.getsignal(signal.SIGTERM)
    try:
        args = ["--task", "tsp_construct", "--run-name", "test_run", "--batch", "test_batch",
                "--base-url", "http://unused", "--model", "scripted", "--budget", "3", "--initial-size", "1"]
        if heldout_failure:
            with pytest.raises(RuntimeError, match="held-out"):
                run.main(args)
        else:
            run.main(args)
    finally:
        signal.signal(signal.SIGTERM, previous)
    text = capsys.readouterr().out
    assert len(tested) == 1 and tested[0].evaluation.fitness == 2
    directory = tmp_path / "experiments_result/traceaad_v19_3/tsp_construct/test_run"
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["status"] == ("stopped" if heldout_failure else "finished")
    assert ("finished_at" in summary) == (not heldout_failure)
    if not heldout_failure:
        output = json.loads(text)
        assert (output["evaluations"], output["model_calls"]) == (3, 5)
    assert (directory / "best_program.py").read_text() == tested[0].code
