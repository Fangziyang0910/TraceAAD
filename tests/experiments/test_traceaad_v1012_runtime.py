"""Regression coverage for source isolation and spawned ACO worker imports."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.traceaad_v10_12 import manual_launch
from experiments.traceaad_v10_12.freeze import verify_runtime


def test_manual_launch_uses_verified_runtime_for_every_session(tmp_path, monkeypatch):
    results = tmp_path / "results"
    runtime = results / "runtime_test"
    runtime.mkdir(parents=True)
    assignments = tmp_path / "assignments.json"
    rows = [{"task": "op_aco", "repeat": 1, "seed": 0,
             "backend": "server1", "run_name": "test_op_v1012_rep1"}]
    assignments.write_text(json.dumps(rows))
    monkeypatch.setattr(manual_launch, "RESULTS_ROOT", results)
    monkeypatch.setattr(manual_launch, "_validate", lambda *a, **kw: None)
    verified = []
    monkeypatch.setattr(manual_launch, "verify_runtime",
                        lambda path: verified.append(path) or "verified-hash")
    launches = []

    def run(command, **kwargs):
        if command[1] == "has-session":
            return subprocess.CompletedProcess(command, 1)
        launches.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(manual_launch.subprocess, "run", run)
    manual_launch.launch("test", assignments, "v1012", 0, do_freeze=False)
    assert verified == [runtime]
    assert len(launches) == 1
    command = launches[0][0]
    assert command[command.index("-c") + 1] == str(runtime)
    assert command[command.index("-e") + 1] == f"PYTHONPATH={runtime}"
    assert launches[0][1]["cwd"] == runtime
    assert launches[0][1]["env"]["PYTHONPATH"] == str(runtime.resolve())
    manifest = json.loads((results / "batch_test.json").read_text())
    assert manifest["source_identity"] == "verified-hash"


def test_runtime_rejects_drift_before_start(tmp_path):
    source = tmp_path / "entry.py"
    source.write_text("answer = 42\n")
    (tmp_path / "runtime_manifest.json").write_text(json.dumps({
        "files": {"entry.py": hashlib.sha256(source.read_bytes()).hexdigest()}}))
    verify_runtime(tmp_path, preflight=False)
    source.write_text("answer = 43\n")
    with pytest.raises(ValueError, match="frozen source changed"):
        verify_runtime(tmp_path, preflight=False)


def test_fresh_search_entry_and_spawn_worker_import():
    root = Path(__file__).resolve().parents[2]
    code = """
import __main__, multiprocessing, os
import experiments.traceaad_v10_12.run as entry
assert 'rand_context' not in vars(entry.build_parser().parse_args([
    '--task', 'op_aco', '--backend', 'server1']))
__main__.__spec__ = entry.__spec__
p = multiprocessing.get_context('spawn').Process(target=os.getpid)
p.start()
p.join(30)
if p.is_alive():
    p.terminate()
    p.join()
    raise RuntimeError('spawn bootstrap timed out')
assert p.exitcode == 0
"""
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True, timeout=45)


def test_direct_search_refuses_mutable_working_tree():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "experiments.traceaad_v10_12.run", "--task", "op_aco"],
        cwd=root, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "search must run from a verified frozen runtime" in result.stderr
