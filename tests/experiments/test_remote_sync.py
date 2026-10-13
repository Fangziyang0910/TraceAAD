import json
import subprocess

from experiments.infra import remote


def test_sync_retries_and_waits_for_every_search_without_pruning(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "results", lambda experiment: ("remote", tmp_path))
    plan = [{"task": "tsp_construct", "run_name": f"run{i}"} for i in range(2)]
    (tmp_path / "batch_test.json").write_text(json.dumps({"plan": plan}))
    calls, sleeps = [], []

    def pull(experiment, batch):
        calls.append((experiment, batch))
        if len(calls) == 1:
            raise subprocess.CalledProcessError(255, "ssh")
        row = plan[len(calls) - 2]
        directory = tmp_path / row["task"] / row["run_name"]
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text(json.dumps({"status": "finished"}))

    def prune(*args, **kwargs):
        raise AssertionError("sync must not prune")

    monkeypatch.setattr(remote, "pull", pull)
    monkeypatch.setattr(remote, "prune", prune)
    monkeypatch.setattr(remote.time, "sleep", sleeps.append)
    remote.main(["sync", "--experiment", "traceaad_v12_0", "--batch", "test"])
    assert calls == [("traceaad_v12_0", "test")] * 3
    assert sleeps == [60, 60]
