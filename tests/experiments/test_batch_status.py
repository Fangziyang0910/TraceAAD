import hashlib
import json
from types import SimpleNamespace

from experiments.infra import batch_status


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def batch(tmp_path):
    manifest = tmp_path / "batch_trial.json"
    write(manifest, {"batch": "trial", "budget_per_run": 1000, "plan": [
        {"task": "tsp_construct", "run_name": "rep1", "run_dir": "/another/host/rep1"}]})
    return manifest, tmp_path / "tsp_construct/rep1"


def freeze(directory):
    code = b"def score():\n    return 1\n"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "best_program.py").write_bytes(code)
    key = hashlib.sha256(code).hexdigest()
    write(directory / "selection.json", {"selected_key": key, "selected_node": 7})
    write(directory / "summary.json", {"status": "finished", "phase": "finished",
                                               "budget": 1000, "budget_used": 1000, "best": {"id": 7, "key": key}})
    return key


def test_manifest_includes_unsynced_runs_without_claiming_they_are_running(tmp_path):
    manifest, _ = batch(tmp_path)
    result = batch_status.collect(manifest)
    assert result["summary"]["unknown"] == 1
    assert result["runs"][0]["budget_used"] is None
    assert result["search_timing"]["eta_seconds"] is None


def test_deliberate_stop_is_not_reported_as_a_failure(tmp_path):
    manifest, directory = batch(tmp_path)
    write(directory / 'summary.json', {'status': 'stopped', 'budget_used': 12,
                                      'budget': 1000, 'stop_reason': 'task replaced'})
    result = batch_status.collect(manifest)
    assert result['runs'][0]['status'] == 'stopped'
    assert result['summary']['stopped'] == 1
    assert result['search_timing']['active_runs'] == 0


def test_tail_ignores_partial_utf8_and_never_falls_back_to_full_journal(tmp_path, monkeypatch):
    monkeypatch.setattr(batch_status, "TAIL_BYTES", 300)
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"kind":"candidate","budget_used":1}\n' + b"x" * 1000 + b"\n" +
                     b'{"kind":"progress","progress":{"attempts":8,"phase":"search"}}\n' +
                     b'{"kind":"candidate","budget_used":9,"idea":"\xe4')
    state, candidate, modified = batch_status.journal_tail(path)
    assert state["attempts"] == 8 and candidate == {} and modified is not None


def test_checkpoint_is_progress_evidence_and_search_complete_is_not_run_finished(tmp_path):
    manifest, directory = batch(tmp_path)
    directory.mkdir(parents=True)
    (directory / "events.jsonl").write_text(json.dumps({"kind": "progress", "progress": {
        "attempts": 1000, "phase": "selection", "started_at": "2026-10-03T00:00:00+09:00", "elapsed": 600}}) + "\n")
    row = batch_status.collect(manifest)["runs"][0]
    assert row["status"] == "recorded" and not row["frozen"]
    assert row["timing"]["state"] == "search_complete"
    assert row["budget_used"] == 1000


def test_failed_and_wrong_program_heldout_are_distinct_from_missing(tmp_path):
    manifest, directory = batch(tmp_path)
    key = freeze(directory)
    heldout = []
    for size, fitness, result_key in [(50, -6.0, key), (100, None, key), (200, -20.0, "wrong")]:
        heldout.append({"task": "tsp_construct", "split": f"eval_{size}", "scale": str(size),
                        "key": result_key, "node_id": 7, "fitness": fitness, "variant": "", "verification": "verified"})
    write(directory / "heldout.json", heldout)
    result = batch_status.collect(manifest)
    row = result["runs"][0]
    assert row["frozen"]
    assert row["heldout"] == {"valid": ["eval_50"], "failed": ["eval_100"], "missing": [], "mismatched": ["eval_200"], "unverified": []}
    assert result["summary"]["ready_for_heldout"] == 0
    (directory / "best_program.py").write_text("changed")
    result = batch_status.collect(manifest)
    assert not result["runs"][0]["frozen"]
    assert result["summary"]["heldout_valid"] == 0


def test_finished_program_with_missing_splits_is_ready(tmp_path):
    manifest, directory = batch(tmp_path)
    freeze(directory)
    assert batch_status.collect(manifest)["summary"]["ready_for_heldout"] == 1


def test_wait_ignores_eta_changes_and_returns_on_new_progress(tmp_path, monkeypatch):
    manifest, _ = batch(tmp_path)
    original = batch_status.collect(manifest)
    later = json.loads(json.dumps(original))
    later["runs"][0]["budget_used"] = 1
    clock = [0]
    monkeypatch.setattr(batch_status.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(batch_status.time, "sleep", lambda duration: clock.__setitem__(0, clock[0] + duration))
    responses = iter([original, {**original, "observed_at": "later"}, later])
    monkeypatch.setattr(batch_status, "collect", lambda path: next(responses))
    result = batch_status.observe(manifest, wait_seconds=30, interval=10)
    assert result["wait_result"] == "changed" and clock[0] == 20


def test_wait_times_out_without_requiring_an_extra_agent_call(tmp_path, monkeypatch):
    manifest, _ = batch(tmp_path)
    clock = [0]
    monkeypatch.setattr(batch_status.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(batch_status.time, "sleep", lambda duration: clock.__setitem__(0, clock[0] + duration))
    assert batch_status.observe(manifest, wait_seconds=5, interval=10)["wait_result"] == "timeout"
    assert clock[0] == 5


def test_remote_query_uses_one_connection_and_quotes_arguments(monkeypatch):
    calls = []
    monkeypatch.setattr(batch_status.subprocess, "run", lambda command, **kwargs:
                        calls.append((command, kwargs)) or SimpleNamespace(returncode=0))
    assert batch_status.main(["--ssh", "B3-server3", "--repo", "/repo with spaces",
                              "--manifest", "results/batch with spaces.json", "--json"]) == 0
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert "cd '/repo with spaces'" in command[-1]
    assert "'results/batch with spaces.json'" in command[-1]
    assert "def journal_tail" in kwargs["input"]
    assert kwargs["timeout"] == 60
