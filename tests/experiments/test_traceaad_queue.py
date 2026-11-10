"""A persisted campaign queue shares capacity with existing workers and their held-out phase."""

import json
from pathlib import Path

import pytest

from experiments.traceaad import queue
from traceaad_base.storage import write_json


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    live, sessions, starts = set(), set(), []

    def manifest(name, count):
        path = tmp_path / name / f"batch_{name}.json"
        rows = [{"task": "tsp_construct", "seed": i, "run_name": f"{name}_r{i}",
                 "session": f"{name}_r{i}", "run_dir": str(path.parent / f"run{i}"),
                 "startup_log": str(path.parent / f"run{i}.log"), "status": "queued", "command": []}
                for i in range(count)]
        write_json(path, {"batch": name, "plan": rows})
        return path, [{**row, "batch": name} for row in rows]

    def launch(row):
        live.add(queue.key(row))
        sessions.add(row["session"])
        starts.append((row["batch"], row["run_name"]))
        write_json(Path(row["run_dir"]) / "summary.json", {"status": "running"})

    def finish(row):
        live.discard(queue.key(row))
        sessions.discard(row["session"])
        write_json(Path(row["run_dir"]) / "summary.json", {"status": "finished"})
        write_json(Path(row["run_dir"]) / "heldout.json", [{"status": "success"}])

    monkeypatch.setattr(queue, "inventory", lambda _keys: (set(live), set(sessions)))
    monkeypatch.setattr(queue, "launch_run", launch)
    return manifest, launch, finish, live, sessions, starts


def test_36_positions_include_existing_runs_and_interleave_waiting_versions(tmp_path, campaign):
    manifest, launch, finish, live, sessions, starts = campaign
    existing, old = manifest("existing", 24)
    first, a = manifest("v21_1", 12)
    second, b = manifest("v22_2", 12)
    third, c = manifest("v24_1", 12)
    for row in old:
        launch(row)
    starts.clear()
    args = dict(tasks=["tsp_construct"], capacity=36, state_path=tmp_path / "queue.json", spacing=0)
    state = queue.step([first, second, third], [existing], **args)
    assert state["counts"] == {"running": 36, "queued": 24}
    assert [batch for batch, _ in starts] == ["v21_1"] * 12
    finish(old[0])
    state = queue.step([first, second, third], [existing], **args)
    assert state["counts"] == {"finished": 1, "running": 36, "queued": 23}
    assert starts[-1][0] == "v22_2"
    finish(old[1])
    state = queue.step([first, second, third], [existing], **args)
    assert state["counts"]["running"] == len(live) == 36
    assert starts[-1][0] == "v24_1"
    before = list(starts)
    queue.step([first, second, third], [existing], **args)
    assert starts == before  # Restart/reconcile never starts existing routes again.


def test_training_completion_does_not_free_the_heldout_position(tmp_path, campaign):
    manifest, launch, finish, live, sessions, starts = campaign
    existing, old = manifest("existing", 1)
    pending, new = manifest("new", 1)
    launch(old[0]); starts.clear()
    directory = Path(old[0]["run_dir"])
    write_json(directory / "summary.json", {"status": "finished"})
    args = dict(tasks=["tsp_construct"], capacity=1, state_path=tmp_path / "queue.json", spacing=0)
    queue.step([pending], [existing], **args)
    assert starts == []
    write_json(directory / "heldout.json", [{"status": "success"}])
    live.clear()
    queue.step([pending], [existing], **args)
    assert starts == []  # A live tmux worker still owns the position.
    sessions.clear()
    state = queue.step([pending], [existing], **args)
    assert starts == [("new", new[0]["run_name"])]
    assert state["counts"] == {"finished": 1, "running": 1}


def test_interrupted_start_is_reconciled_and_missing_heldout_is_blocked(tmp_path, campaign):
    manifest, launch, finish, live, sessions, starts = campaign
    pending, rows = manifest("pending", 2)
    saved = json.loads(pending.read_text()); saved["plan"][0]["status"] = "starting"; write_json(pending, saved)
    args = dict(tasks=["tsp_construct"], capacity=1, state_path=tmp_path / "queue.json", spacing=0)
    queue.step([pending], [], **args)
    assert starts == [("pending", rows[0]["run_name"])]
    write_json(Path(rows[0]["run_dir"]) / "summary.json", {"status": "finished"})
    live.clear(); sessions.clear()
    state = queue.step([pending], [], **args)
    assert state["runs"][rows[0]["run_name"]]["status"] == "blocked"
    assert starts[-1] == ("pending", rows[1]["run_name"])
    assert len(starts) == 2
    with pytest.raises(ValueError, match="settings changed"):
        queue.step([pending], [], **{**args, "capacity": 2})


def test_excluded_tasks_and_duplicate_manifests_never_launch(tmp_path, campaign):
    manifest, launch, finish, live, sessions, starts = campaign
    pending, rows = manifest("pending", 1)
    args = dict(tasks=["op_aco"], capacity=1, state_path=tmp_path / "queue.json", spacing=0)
    with pytest.raises(ValueError, match="excluded task"):
        queue.step([pending], [], **args)
    with pytest.raises(ValueError, match="distinct"):
        queue.step([pending], [pending], **args)
    assert starts == []


def test_failed_start_recounts_before_launching_another_route(tmp_path, campaign, monkeypatch):
    manifest, launch, finish, live, sessions, starts = campaign
    pending, rows = manifest("pending", 2)

    def interrupted(row):
        launch(row)
        raise FileExistsError("Concurrent tmux start")

    monkeypatch.setattr(queue, "launch_run", interrupted)
    args = dict(tasks=["tsp_construct"], capacity=1, state_path=tmp_path / "queue.json", spacing=0)
    queue.step([pending], [], **args)
    assert len(starts) == len(live) == 1
    state = queue.step([pending], [], **args)
    assert state["counts"] == {"running": 1, "queued": 1}
    assert len(starts) == 1


def test_queued_and_deferred_statuses_are_explicit(tmp_path):
    from experiments.infra.batch_status import run_status

    row = {"task": "tsp_construct", "run_name": "queued", "status": "queued"}
    assert run_status(tmp_path, row, 1000, 0)["status"] == "queued"
    row["status"] = "deferred"
    write_json(tmp_path / "tsp_construct/queued/summary.json", {"status": "stopped"})
    assert run_status(tmp_path, row, 1000, 0)["status"] == "deferred"
