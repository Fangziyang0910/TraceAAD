import io
import json
import subprocess

from experiments.infra import watch_server4
from experiments.infra.watch_server4 import active_batches, batch_issues, lag_issues


def test_watch_distinguishes_live_search_from_missing_session_and_test():
    plan = [{"run_name": "one", "session": "session-one"}]
    row = {"name": "one", "status": "recorded", "frozen": False,
           "heldout": {"failed": [], "mismatched": [], "unverified": [], "missing": ["test"]}}
    assert batch_issues({"runs": [row]}, plan, {"session-one"}) == []
    assert "搜索会话已退出" in batch_issues({"runs": [row]}, plan, set())[0]
    row.update(status="finished", frozen=True)
    assert batch_issues({"runs": [row]}, plan, {"session-one"}) == []
    assert "主测试缺失" in batch_issues({"runs": [row]}, plan, set())[0]


def test_active_batches_are_those_on_server4_with_a_primary_run_not_complete_here(monkeypatch, tmp_path):
    # b_rerun is the longest prefix of its run; a's run is complete; c's manifest is not on server4.
    monkeypatch.setattr(watch_server4, "RESULTS", tmp_path)
    runs = {"m": ["t/a_t_m_rep1", "t/b_rerun_t_m_rep1", "t/c_t_m_rep1"]}
    (tmp_path / "primary_runs.json").write_text(json.dumps({"methods": runs}))
    for name, summary in (("a_t_m_rep1", {"status": "search_complete"}), ("c_t_m_rep1", {"status": "finished"})):
        (tmp_path / "m/t" / name).mkdir(parents=True)
        (tmp_path / "m/t" / name / "summary.json").write_text(json.dumps(summary))
    assert active_batches({"m": ["a", "b", "b_rerun"]}) == [("m", "b_rerun")]


def test_the_local_copy_counts_as_behind_by_candidates_or_by_a_finished_run():
    there = {"runs": [{"name": "one", "status": "recorded", "budget_used": 40},
                      {"name": "two", "status": "finished", "budget_used": 1000}]}
    here = {"runs": [{"name": "one", "status": "recorded", "budget_used": 21},
                     {"name": "two", "status": "recorded", "budget_used": 1000}]}
    assert lag_issues(there, here) == ["two: 本机结果落后于 server4"]
    here["runs"][0]["budget_used"] = 20
    assert len(lag_issues(there, here)) == 2


def test_watch_notifies_only_new_problems_and_retries_failed_notifications(monkeypatch, tmp_path):
    monkeypatch.setattr(watch_server4, "STATE", tmp_path / "status.json")
    monkeypatch.setattr(watch_server4, "remote_manifests", dict)
    monkeypatch.setattr(watch_server4, "build_opener", lambda _: type("Opener", (), {
        "open": lambda *args, **kwargs: io.BytesIO(b'{"status":"ok"}')})())

    def command(args):
        if args[0] == "systemctl":
            return "[]"
        if args[-1].startswith("curl "):
            raise subprocess.CalledProcessError(1, args)
        return "session-one"

    monkeypatch.setattr(watch_server4, "command", command)
    notifications = []

    def notify(args, **kwargs):
        notifications.append(args)
        if len(notifications) == 1:
            raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(watch_server4.subprocess, "run", notify)
    watch_server4.check()
    assert json.loads(watch_server4.STATE.read_text())["notified"] == []
    watch_server4.check()
    watch_server4.check()
    assert len(notifications) == 2
    assert json.loads(watch_server4.STATE.read_text())["notified"] == ["server4 无法访问本机模型"]
