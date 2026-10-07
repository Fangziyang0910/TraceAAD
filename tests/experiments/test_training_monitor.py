import json
import math
from datetime import datetime
from traceaad.common.storage import Programs

from experiments.monitor import ResultsMonitor


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_run(tmp_path):
    results = tmp_path / "results"
    run_name = "batch_tsp_v1013_rep1"
    write_json(results / "batch_batch.json", {
        "method": "v1013_three_pool", "batch": "batch",
        "created_at": "2026-09-25T10:00:00", "updated_at": "2026-09-25T11:00:00",
        "plan": [{"task": "tsp_construct", "repeat": 1, "seed": 0,
                  "backend": "server1", "run_name": run_name, "status": "running"}]})
    run = results / "tsp_construct" / run_name
    write_json(run / "run_config.json", {"task": "tsp_construct", "budget": 4,
               "budget_axis": "评价次数", "method_params": {"budget": 4}})
    append_records(run / "events.jsonl", [
        candidate(1, 12.0, operator="Init", idea="first", code="def f(): return 1"),
        candidate(2, 10.0, operator="Refine", idea="best", code="def f(): return 2"),
        candidate(3, None, operator="Tune")])
    return results, run_name


def test_overview_reads_v1013_progress(tmp_path):
    results, _ = make_run(tmp_path)
    state = ResultsMonitor(results.parent, results.name).overview("results")

    assert state["batch"] == "results"
    assert {key: state["summary"][key] for key in ("runs", "finished", "running", "queued", "blocked", "budget_used", "budget", "valid_nodes")} == {
        "runs": 1,
        "finished": 0,
        "running": 1,
        "queued": 0,
        "blocked": 0,
        "budget_used": 3,
        "budget": 4,
        "valid_nodes": 2,
    }
    assert state["tasks"][0]["runs"][0]["best_value"] == 10.0


def test_active_run_ignores_stale_error_summary(tmp_path):
    results, run_name = make_run(tmp_path)
    write_json(results / "tsp_construct" / run_name / "summary.json", {
        "status": "error",
        "budget": 4,
        "budget_used": 1,
        "num_nodes": 1,
        "best": {"fitness": 12.0},
        "error": "old connection failure",
    })

    run = ResultsMonitor(results.parent, results.name).overview("results")["tasks"][0]["runs"][0]

    assert run["status"] == "running"
    assert run["budget_used"] == 3
    assert run["valid_nodes"] == 2
    assert run["best_value"] == 10.0
    assert run["error"] is None


def test_batch_list_exposes_experiment_with_legacy_manifests(tmp_path):
    results, _ = make_run(tmp_path)
    write_json(results / "batch_old.json", {
        "method": "v1013_three_pool",
        "batch": "old",
        "created_at": "2026-09-24T10:00:00",
        "plan": [{"task": "tsp_construct"}],
    })

    assert [item["id"] for item in ResultsMonitor(results.parent, results.name).batches()] == ["results"]


def test_run_detail_builds_minimization_curve_and_best_program(tmp_path):
    results, run_name = make_run(tmp_path)
    detail = ResultsMonitor(results.parent, results.name).run_detail("results", "tsp_construct", run_name)

    assert detail is not None
    assert [{k: p[k] for k in ("evaluation", "value")} for p in detail["curve"]] == [
        {"evaluation": 1, "value": 12.0},
        {"evaluation": 2, "value": 10.0},
        {"evaluation": 3, "value": 10.0},
    ]
    assert detail["operators"] == {"Init": 1, "Refine": 1, "Tune": 1}
    assert detail["best"]["idea"] == "best"
    assert detail["best"]["value"] == 10.0


def test_finished_run_uses_summary_and_journal(tmp_path):
    results, run_name = make_run(tmp_path)
    run_dir = results / "tsp_construct" / run_name
    best = next(record["program"] for record in reversed(
        [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()])
        if record.get("program"))
    write_json(run_dir / "summary.json", {"status": "finished", "budget": 4, "budget_used": 3,
                          "num_nodes": 2, "best": best})

    monitor = ResultsMonitor(results.parent, results.name)
    overview = monitor.overview("results")
    detail = monitor.run_detail("results", "tsp_construct", run_name)
    assert overview["summary"]["budget_used"] == 3
    assert overview["summary"]["valid_nodes"] == 2
    assert overview["summary"]["finished"] == 1
    assert detail["best"]["code"] == Programs(run_dir).get(best["key"])


def test_unified_monitor_lists_experiments_and_reads_summary(tmp_path):
    root = tmp_path / "results"
    results, run_name = make_run(tmp_path)
    run_dir = results / "tsp_construct" / run_name
    write_json(run_dir / "summary.json", {
        "status": "finished", "budget": 4, "budget_used": 3,
        "num_nodes": 2, "best": next(r["program"] for r in reversed(
            [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]) if r.get("program")),
    })
    target = root / "comparison" / "tsp_construct"
    target.mkdir(parents=True)
    run_dir.rename(target / run_name)

    monitor = ResultsMonitor(root, "comparison")
    assert monitor.batches() == [{"id": "comparison", "label": "comparison"}]
    overview = monitor.overview("comparison")
    assert overview["summary"]["finished"] == 1
    assert overview["tasks"][0]["runs"][0]["best_value"] == 10
    assert monitor.run_detail("comparison", "tsp_construct", run_name)["best"]["code"] == "def f(): return 2"
    assert monitor.run_detail("comparison", "tsp_construct", "missing") is None


def test_historical_budget_and_incomplete_node(tmp_path):
    run = tmp_path / "traceaad_v9_19" / "tsp_construct" / "rep1"
    write_json(run / "run_config.json", {"task": "tsp_construct", "repeat": 1,
               "budget": 1000, "budget_axis": "预算槽位"})
    write_json(run / "summary.json", {"status": "finished", "budget": 1000,
               "budget_used": 1000, "evaluation_calls": 1097, "best": None})
    append_records(run / "events.jsonl", [candidate(1, 9), candidate(2, 7)])
    monitor = ResultsMonitor(tmp_path, "traceaad_v9_19")
    row = monitor.overview("traceaad_v9_19")["tasks"][0]["runs"][0]
    assert row["budget_used"] == row["budget"] == 1000
    curve = monitor.run_detail("traceaad_v9_19", "tsp_construct", "rep1")["curve"]
    assert [{k: p[k] for k in ("evaluation", "value")} for p in curve] == [
        {"evaluation": 1, "value": 9.0}, {"evaluation": 2, "value": 7.0}]
    write_json(run / "summary.json", {"status": "unknown"})
    (run / "events.jsonl").unlink()
    append_records(run / "events.jsonl", [candidate(1, 8, code="first"), candidate(2, 7, code="second")])
    detail = monitor.run_detail("traceaad_v9_19", "tsp_construct", "rep1")
    assert detail["status"] == "unknown"
    assert detail["best"]["code"] == "second"


def test_missing_final_summary_uses_recorded_evaluations(tmp_path):
    run = tmp_path / "traceaad_v9_7" / "tsp_construct" / "rep1"
    write_json(run / "run_config.json", {"task": "tsp_construct", "budget": 10,
               "budget_axis": "评价次数"})
    write_json(run / "summary.json", {"status": "unknown"})
    append_records(run / "events.jsonl", [candidate(1, 9), candidate(3, 8, budget_used=2)])
    row = ResultsMonitor(tmp_path).overview("traceaad_v9_7")["tasks"][0]["runs"][0]
    assert (row["status"], row["budget_used"], row["valid_nodes"]) == ("unknown", 2, 2)


def append_records(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for record in records:
            record = dict(record)
            if record.get("program"):
                meta = dict(record["program"])
                meta["key"] = Programs(path.parent).add(meta.pop("code"))
                record["program"] = meta
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def candidate(index, fitness, **metadata):
    valid = fitness is not None and math.isfinite(fitness)
    fitness = fitness if valid else None
    code = metadata.pop("code", f"def f(): return {index}")
    operator = metadata.pop("operator", "Refine")
    idea = metadata.pop("idea", "")
    attempt = {"id": index, "action": operator, "idea": idea,
               "program_id": index if valid else None, "parent_id": metadata.pop("parent_id", None),
               "channel": metadata.pop("channel", None), "call_ids": []}
    program = {"id": index, "fitness": fitness, "valid": valid, "action": operator,
               "idea": idea, "code": code} if valid else None
    return {"kind": "candidate", "candidate_id": index, "budget_used": index,
            "fitness": fitness, "valid": valid, "node_id": index if valid else None,
            "status": "valid" if valid else "evaluation_failed", "operator": operator,
            "attempt": attempt, "program": program, "ts": datetime.now().astimezone().isoformat(),
            "progress": {"attempts": index, "phase": "search"}, **metadata}


def test_breakthroughs_use_incumbent_not_parent_and_keep_operator_provenance(tmp_path):
    from experiments.infra.monitor_history import TrainingHistory
    records = [candidate(1, 12, operator="Init"),
               candidate(2, 10, operator="Refine · Transfer", channel="trial", parent_id=1, idea="真实修改"), candidate(3, 11), candidate(4, 10),
               candidate(5, None), candidate(6, 9, operator="TRACE_RECHECK"), candidate(7, None)]
    append_records(tmp_path / "events.jsonl", records)
    points, recent, operators, outcomes = TrainingHistory(tmp_path, minimize=True).read()
    assert [p["evaluation"] for p in points] == [1, 2, 6, 7]
    assert [p["fitness"] for p in points] == [12, 10, 9, 9]
    assert [p["value"] for p in points] == [12, 10, 9, 9]
    assert [p["kind"] for p in points] == ["initial", "breakthrough", "breakthrough", "progress"]
    assert points[0]["gain"] is None
    assert points[1]["gain"] == 2 and points[2]["gain"] == 1
    assert points[1]["operator"] == "Refine · Transfer"
    assert points[1]["channel"] == "trial" and points[1]["parent_id"] == 1
    assert points[1]["idea"] == "真实修改"
    assert "operator" not in points[-1]  # A failed tail must not inherit a successful operator.
    assert recent[0]["fitness"] is None
    assert operators["Refine · Transfer"] == 1 and outcomes["evaluation_failed"] == 2


def test_all_breakthroughs_survive_beyond_old_240_point_limit(tmp_path):
    from experiments.infra.monitor_history import TrainingHistory
    append_records(tmp_path / "events.jsonl", [candidate(i, -(i / 100)) for i in range(1, 401)])
    points, _, _, _ = TrainingHistory(tmp_path, minimize=False).read()
    assert len(points) == 400
    assert [p["candidate"] for p in points] == list(range(1, 401))
    assert sum(p["kind"] == "breakthrough" for p in points) == 399
    assert all(p["fitness"] == p["value"] for p in points)


def test_live_history_retries_partial_utf8_tail_and_does_not_recount(tmp_path):
    from experiments.infra.monitor_history import TrainingHistory
    path = tmp_path / "events.jsonl"
    append_records(path, [candidate(1, -1)])
    reader = TrainingHistory(tmp_path, minimize=False)
    first = reader.read()
    offset = reader.offset
    payload = (json.dumps(candidate(2, -2, idea="改进", program=None), ensure_ascii=False) + "\n").encode()
    cut = payload.index("改".encode()) + 1
    with path.open("ab") as handle:
        handle.write(payload[:cut])
    assert reader.read() is first
    assert reader.offset == offset
    with path.open("ab") as handle:
        handle.write(payload[cut:])
    second = reader.read()
    assert [p["candidate"] for p in second[0]] == [1, 2]
    assert second[2] == {"Refine": 2}
    assert reader.read() is second
    assert reader.offset == path.stat().st_size


def test_history_cache_resets_after_journal_replacement(tmp_path):
    from experiments.infra.monitor_history import TrainingHistory
    path = tmp_path / "events.jsonl"
    append_records(path, [candidate(1, -1), candidate(2, -2)])
    reader = TrainingHistory(tmp_path, minimize=False)
    assert len(reader.read()[0]) == 2
    replacement = tmp_path / "replacement.jsonl"
    append_records(replacement, [candidate(1, -9)])
    replacement.replace(path)
    assert [p["fitness"] for p in reader.read()[0]] == [-9]


def test_invalid_scores_never_create_breakthroughs(tmp_path):
    from experiments.infra.monitor_history import TrainingHistory
    append_records(tmp_path / "events.jsonl", [candidate(1, None), candidate(2, -(float("nan"))),
                   candidate(3, -(float("inf"))), candidate(4, 0), candidate(5, 0), candidate(6, 1)])
    points, _, _, _ = TrainingHistory(tmp_path, minimize=False).read()
    assert [(p["evaluation"], p["kind"]) for p in points] == [(4, "initial"), (6, "progress")]
    assert points[0]["fitness"] == 0


def test_overview_exposes_same_curve_as_detail_with_candidate_axis(tmp_path):
    run = tmp_path / "traceaad_v10_14" / "op_aco" / "rep1"
    write_json(run / "run_config.json", {"method": "v1014", "task": "op_aco", "repeat": 1})
    write_json(run / "summary.json", {"status": "running", "budget": 1000,
               "best": None})
    append_records(run / "events.jsonl", [candidate(1, -1), candidate(2, -3)])
    monitor = ResultsMonitor(tmp_path)
    row = monitor.overview("traceaad_v10_14")["tasks"][0]["runs"][0]
    detail = monitor.run_detail("traceaad_v10_14", "op_aco", "rep1")
    # The overview carries a slim projection of the detail curve.
    fields = ("evaluation", "fitness", "value", "kind", "gain", "candidate", "operator")
    assert row["curve"] == [{k: p[k] for k in fields if p.get(k) is not None} for p in detail["curve"]]
    assert row["x_label"] == detail["x_label"] == "候选尝试"
    assert row["curve"][-1]["gain"] == 2


def test_live_path_aliases_do_not_duplicate_runs_or_experiments(tmp_path):
    results, name = make_run(tmp_path)
    run = results / "tsp_construct" / name
    old_experiment = results.parent / "old_experiment"
    old_experiment.symlink_to(results, target_is_directory=True)
    old_run = run.parent / "old_run"
    old_run.symlink_to(run, target_is_directory=True)
    monitor = ResultsMonitor(results.parent)
    assert monitor.batches() == [{"id": "results", "label": "results"}]
    before = monitor.state_signature("results")
    append_records(old_experiment / "tsp_construct" / "old_run" / "events.jsonl", [
        candidate(4, 8.0, operator="Refine", idea="continued", code="def f(): return 4")])
    state = monitor.overview("results")
    assert state["summary"]["runs"] == 1 and state["summary"]["budget_used"] == 4
    assert state["tasks"][0]["runs"][0]["name"] == name
    assert state["tasks"][0]["runs"][0]["best_value"] == 8.0
    assert monitor.state_signature("results") != before
