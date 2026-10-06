from datetime import datetime, timedelta, timezone
import json
import os
import time

import pytest

from experiments.infra.monitor_history import TrainingHistory
from experiments.infra.monitor_timing import batch_timing, search_timing, timestamp
from experiments.monitor import ResultsMonitor
from tests.experiments.test_training_monitor import append_records, candidate, make_run, write_json


START = "2026-09-27T12:00:00+09:00"
LAST = "2026-09-27T12:10:00+09:00"
NOW = timestamp(LAST)


def run(used=10, budget=100, status="running"):
    return {"budget_used": used, "budget": budget, "status": status, "valid_nodes": 2}


def summary(**updates):
    return {"started_at": START, "finished_at": LAST, "phase": "search", **updates}


def test_average_counts_failed_budget_units_and_running_time_keeps_advancing():
    first = search_timing(run(), summary(), unit="候选", now=NOW)
    assert first["rate_per_minute"] == 1
    assert first["eta_seconds"] == 5400  # 90 remaining, not 98 valid programs.
    assert timestamp(first["eta_at"]) == NOW + 5400
    later = search_timing(run(), summary(), unit="候选", now=NOW + 60)
    assert later["elapsed_seconds"] == 660
    assert later["eta_seconds"] == 5940


def test_native_elapsed_excludes_pause_and_includes_current_wait():
    snapshot = {"completed": 10, "elapsed": 300, "started_at": START}
    timing = search_timing(run(), summary(), snapshot, now=NOW + 60)
    assert timing["basis"] == "active_time"
    assert timing["elapsed_seconds"] == 360
    assert timing["rate_per_minute"] == pytest.approx(10 / 6)
    assert timing["eta_seconds"] == 3240
    # A concurrent append must not divide an older count by a newer elapsed value.
    snapshot["completed"] = 11
    timing = search_timing(run(), summary(), snapshot, now=NOW)
    assert timing["basis"] == "wall_time"
    assert timing["elapsed_seconds"] == 600


@pytest.mark.parametrize("updates,status,expected", [
    ({"phase": "selection"}, "running", "search_complete"),
    ({"phase": "freeze"}, "running", "search_complete"),
    ({"phase": "selection_failed"}, "blocked", "search_complete"),
    ({}, "finished", "finished"),
])
def test_search_ends_before_selection_even_with_unspent_budget(updates, status, expected):
    snapshot = {"completed": 10, "elapsed": 500, "started_at": START}
    result = search_timing(run(status=status), summary(**updates), snapshot, now=NOW + 7200)
    assert result["state"] == expected
    assert result["eta_seconds"] == 0 and result["eta_at"] is None
    assert result["elapsed_seconds"] == 500  # Selection time must not dilute search speed.


def test_budget_exhaustion_does_not_claim_entire_run_finished():
    timing = search_timing(run(100), summary(), now=NOW)
    assert timing["state"] == "search_complete"
    assert timing["eta_seconds"] == 0


@pytest.mark.parametrize("status", ["queued", "blocked", "unknown", "paused"])
def test_inactive_runs_never_get_a_completion_estimate(status):
    result = search_timing(run(status=status), summary(), now=NOW + 3600)
    assert result["state"] == "inactive"
    assert result["eta_seconds"] is result["eta_at"] is None


@pytest.mark.parametrize("used,age,expected", [(0, 0, "warming_up"), (2, 0, "warming_up"),
                                             (10, 960, "stale"), (1, 1800, "warming_up")])
def test_warmup_and_adaptive_staleness(used, age, expected):
    result = search_timing(run(used), summary(), now=NOW + age)
    assert result["state"] == expected
    assert result["rate_per_minute"] is None and result["eta_seconds"] is None


@pytest.mark.parametrize("data", [{}, {"started_at": "invalid", "finished_at": LAST},
                                   {"started_at": LAST, "finished_at": LAST},
                                   {"started_at": START, "finished_at": "2099-01-01T00:00:00Z"}])
def test_missing_or_invalid_timestamps_are_not_zero_eta(data):
    timing = search_timing(run(), data, now=NOW)
    assert timing["state"] == "unavailable"
    assert timing["eta_seconds"] is None


def test_mixed_timezone_offsets_and_naive_server_time():
    old = os.environ.get("TZ")
    try:
        os.environ["TZ"] = "Asia/Tokyo"
        time.tzset()
        assert timestamp("2026-09-27T12:00:00") == timestamp(START)
        result = search_timing(run(), summary(started_at="2026-09-27T03:00:00Z"), now=NOW)
        assert result["rate_per_minute"] == 1
        assert datetime.fromisoformat(result["eta_at"]).utcoffset().total_seconds() == 0
    finally:
        if old is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old
        time.tzset()


def test_batch_eta_waits_for_slowest_run_and_requires_full_coverage():
    rows = [run(), run(50)]
    for row in rows:
        row["timing"] = search_timing(row, summary(), unit="候选", now=NOW)
    batch = batch_timing(rows)
    assert batch["rate_per_minute"] == 6
    assert batch["eta_seconds"] == 5400
    assert batch["eta_at"] == rows[0]["timing"]["eta_at"]
    waiting = run(status="queued")
    waiting["timing"] = search_timing(waiting, {}, unit="候选", now=NOW)
    incomplete = batch_timing([*rows, waiting])
    assert incomplete["state"] == "partial"
    assert incomplete["eta_runs"] == 2 and incomplete["pending_runs"] == 3
    assert incomplete["eta_seconds"] is None
    # Different budget meanings must never be summed into a fictitious speed.
    rows[1]["timing"]["unit"] = "评价"
    assert batch_timing(rows)["rate_per_minute"] is None
    done = run(status="finished")
    done["timing"] = search_timing(done, summary(), unit="候选", now=NOW)
    assert batch_timing([done])["eta_seconds"] == 0
    assert batch_timing([])["eta_seconds"] is None


def test_native_clock_uses_completed_candidate_and_latest_checkpoint_phase(tmp_path):
    path = tmp_path / "events.jsonl"
    append_records(path, [{**candidate(10, None), "ts": LAST, "progress": {"attempts": 10, "elapsed": 600, "started_at": START, "phase": "search"}}])
    reader = TrainingHistory(tmp_path, minimize=False)
    assert reader.timing_snapshot()["completed"] == 10
    # Charged generation is not a completed candidate. Decode one final checkpoint only.
    checkpoints = [{"kind": "progress", "ts": LAST, "progress": {"phase": phase, "elapsed": 700,
                    "attempts": 10}} for phase in ("search", "freeze", "selection")]
    with path.open("a") as handle:
        for record in checkpoints:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    snapshot = reader.timing_snapshot()
    assert snapshot["completed"] == 10 and snapshot["elapsed"] == 600
    assert snapshot["phase"] == "selection"
    assert search_timing(run(), summary(), snapshot, now=NOW)["state"] == "search_complete"
    replacement = tmp_path / "new.jsonl"
    append_records(replacement, [candidate(1, 2)])
    replacement.replace(path)
    assert reader.timing_snapshot()["elapsed"] is None
    assert reader.timing_snapshot()["completed"] == 1


def test_api_overview_and_detail_use_native_elapsed(tmp_path, monkeypatch):
    monkeypatch.setattr("experiments.monitor.time.time", lambda: NOW)
    directory = tmp_path / "traceaad_v10_14" / "op_aco" / "rep1"
    write_json(directory / "run_config.json", {"method": "v1014", "task": "op_aco", "repeat": 1})
    write_json(directory / "summary.json", {**summary(), **run(), "best": None})
    append_records(directory / "events.jsonl", [{**candidate(10, None), "ts": LAST, "progress": {
        "attempts": 10, "elapsed": 300, "started_at": START, "phase": "search"}}])
    monitor = ResultsMonitor(tmp_path)
    state = monitor.overview("traceaad_v10_14")
    timing = state["tasks"][0]["runs"][0]["timing"]
    assert timing["unit"] == "候选" and timing["rate_per_minute"] == 2
    assert timing == monitor.run_detail("traceaad_v10_14", "op_aco", "rep1")["timing"]
    assert state["summary"]["timing"]["eta_seconds"] == timing["eta_seconds"]


@pytest.mark.parametrize("method,experiment", [
    ("v1015", "traceaad_v10_15"), ("v1016", "traceaad_v10_16")])
@pytest.mark.parametrize("clock_offset_hours", [-1, 8])
@pytest.mark.parametrize("log_age", [0, 1000])
def test_copied_checkpoint_eta_uses_preserved_file_time(
        tmp_path, monkeypatch, method, experiment, clock_offset_hours, log_age):
    monkeypatch.setattr("experiments.monitor.time.time", lambda: NOW)
    directory = tmp_path / experiment / "op_aco" / "rep1"
    write_json(directory / "run_config.json", {
        "method": method, "task": "op_aco", "budget": 100, "method_params": {"budget": 100}})
    # Copying preserves the file timestamp, but a server's naive clock can be
    # either ahead of or behind the viewer's local timezone.
    foreign_start = (datetime.fromtimestamp(NOW - 600 - log_age)
                     + timedelta(hours=clock_offset_hours)).isoformat()
    journal = directory / "events.jsonl"
    append_records(journal, [candidate(10, 1, ts=datetime.fromtimestamp(NOW-log_age, timezone.utc).isoformat(),
        progress={"attempts": 10, "elapsed": 600, "started_at": foreign_start, "phase": "search"})])
    os.utime(journal, (NOW - log_age, NOW - log_age))
    state = ResultsMonitor(tmp_path).overview(experiment)
    timing = state["tasks"][0]["runs"][0]["timing"]
    if log_age:
        assert timing["state"] == "stale" and timing["eta_seconds"] is None
    else:
        assert timing["state"] == "estimated"
        assert timing["rate_per_minute"] == 1
        assert timing["eta_seconds"] == 5400
        assert state["summary"]["timing"]["eta_seconds"] == 5400


def test_legacy_v1013_reports_evaluations_per_minute(tmp_path, monkeypatch):
    monkeypatch.setattr("experiments.monitor.time.time", lambda: NOW)
    root, name = make_run(tmp_path)
    write_json(root / "tsp_construct" / name / "summary.json", summary())
    journal = root / "tsp_construct" / name / "events.jsonl"
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    for record in records:
        record["ts"] = LAST
        record["progress"].update(elapsed=600, started_at=START)
    journal.write_text("\n".join(json.dumps(r) for r in records)+"\n")
    row = ResultsMonitor(root.parent, root.name).overview(root.name)["tasks"][0]["runs"][0]
    assert row["timing"]["unit"] == "评价"
    assert row["timing"]["rate_per_minute"] == .3
