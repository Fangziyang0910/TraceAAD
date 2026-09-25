"""Regression coverage for worktree launches and spawned ACO worker imports."""

import json
from pathlib import Path
import subprocess

from experiments.traceaad_v10_12 import manual_launch


def test_manual_launch_uses_uv_from_worktree(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    results = tmp_path / "results"
    assignments = tmp_path / "assignments.json"
    rows = [{"task": "op_aco", "repeat": 1, "seed": 0,
             "backend": "server1", "run_name": "test_op_v1012_rep1"}]
    assignments.write_text(json.dumps(rows))
    monkeypatch.setattr(manual_launch, "ROOT", root)
    monkeypatch.setattr(manual_launch, "RESULTS_ROOT", results)
    monkeypatch.setattr(manual_launch, "_validate", lambda *a, **kw: None)
    launches = []

    def run(command, **kwargs):
        if command[1] == "has-session":
            return subprocess.CompletedProcess(command, 1)
        launches.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(manual_launch.subprocess, "run", run)
    manual_launch.launch("test", assignments, "v1012", 0)
    assert len(launches) == 1
    command, kwargs = launches[0]
    assert command[command.index("-c") + 1] == str(root)
    uv_index = command.index("uv")
    assert command[uv_index:uv_index + 5] == [
        "uv", "run", "python", "-m", "experiments.traceaad_v10_12.run"]
    assert kwargs["cwd"] == root
    manifest = json.loads((results / "batch_test.json").read_text())
    assert manifest["plan"][0]["run_name"] == rows[0]["run_name"]


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
    subprocess.run(["uv", "run", "python", "-c", code],
                   cwd=root, check=True, timeout=45)


def test_direct_search_entry_is_available_from_worktree():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["uv", "run", "python", "-m", "experiments.traceaad_v10_12.run", "--help"],
        cwd=root, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0
    assert "--n-profile-cards" in result.stdout
