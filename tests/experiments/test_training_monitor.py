import json

from experiments.monitor import ResultsMonitor, V1013Monitor
from traceaad.v10_13.storage import RunStorage


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_run(tmp_path):
    results = tmp_path / "results"
    run_name = "batch_tsp_v1013_rep1"
    row = {
        "task": "tsp_construct",
        "repeat": 1,
        "seed": 0,
        "backend": "server1",
        "run_name": run_name,
        "status": "running",
    }
    write_json(results / "batch_batch.json", {
        "method": "v1013_three_pool",
        "batch": "batch",
        "created_at": "2026-09-25T10:00:00",
        "updated_at": "2026-09-25T11:00:00",
        "plan": [row],
    })
    run_dir = results / "tsp_construct" / run_name
    write_json(run_dir / "run_config.json", {"method_params": {"budget": 4}})
    events = [
        {"candidate_id": 1, "budget_used": 1, "operator": "Init", "status": "ok", "fitness": -12.0},
        {"candidate_id": 2, "budget_used": 2, "operator": "Refine", "status": "ok", "fitness": -10.0},
        {"candidate_id": 3, "budget_used": 3, "operator": "Tune", "status": "eval_failed", "fitness": None},
    ]
    nodes = [
        {"id": 0, "fitness": -12.0, "operator": "Init", "idea": "first", "code": "def f(): return 1"},
        {"id": 1, "fitness": -10.0, "operator": "Refine", "idea": "best", "code": "def f(): return 2"},
    ]
    storage = RunStorage(run_dir)
    for index, event in enumerate(events):
        storage.commit_candidate(event, nodes[index] if index < len(nodes) else None,
                                 {"candidate_count": index + 1, "budget_used": index + 1})
    return results, run_name


def test_overview_reads_v1013_progress(tmp_path):
    results, _ = make_run(tmp_path)
    state = V1013Monitor(results).overview("batch")

    assert state["batch"] == "batch"
    assert {key: value for key, value in state["summary"].items() if key != "timing"} == {
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
    write_json(results / "tsp_construct" / run_name / "logs/run_summary.json", {
        "status": "error",
        "budget": 4,
        "budget_used": 1,
        "num_nodes": 1,
        "best": {"fitness": -12.0},
        "error": "old connection failure",
    })

    run = V1013Monitor(results).overview("batch")["tasks"][0]["runs"][0]

    assert run["status"] == "running"
    assert run["budget_used"] == 3
    assert run["valid_nodes"] == 2
    assert run["best_value"] == 10.0
    assert run["error"] is None


def test_batch_list_only_exposes_latest_v1013_batch(tmp_path):
    results, _ = make_run(tmp_path)
    write_json(results / "batch_old.json", {
        "method": "v1013_three_pool",
        "batch": "old",
        "created_at": "2026-09-24T10:00:00",
        "plan": [{"task": "tsp_construct"}],
    })

    assert [item["id"] for item in V1013Monitor(results).batches()] == ["batch"]


def test_run_detail_builds_minimization_curve_and_best_program(tmp_path):
    results, run_name = make_run(tmp_path)
    detail = V1013Monitor(results).run_detail("batch", "tsp_construct", run_name)

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
    storage = RunStorage(run_dir)
    best = storage.records("nodes")[-1]
    storage.save_summary({"status": "finished", "budget": 4, "budget_used": 3,
                          "num_nodes": 2, "best": best})

    monitor = V1013Monitor(results)
    overview = monitor.overview("batch")
    detail = monitor.run_detail("batch", "tsp_construct", run_name)
    assert overview["summary"]["budget_used"] == 3
    assert overview["summary"]["valid_nodes"] == 2
    assert overview["summary"]["finished"] == 1
    assert detail["best"]["code"] == best["code"]


def test_unified_monitor_lists_experiments_and_reads_summary(tmp_path):
    root = tmp_path / "results"
    results, run_name = make_run(tmp_path)
    run_dir = results / "tsp_construct" / run_name
    write_json(run_dir / "logs/run_summary.json", {
        "status": "finished", "budget": 4, "budget_used": 3,
        "num_nodes": 2, "best": {"fitness": -10, "code": "def f(): return 2"},
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
    write_json(run / "run_config.json", {
        "task": "tsp_construct", "repeat": 1, "method_params": {"budget": 1000},
    })
    write_json(run / "logs/run_summary.json", {
        "status": "finished", "budget_slots": 1000,
        "evaluator_call_count": 1097, "best_score": -6.0,
    })
    (run / "search.jsonl").write_text("\n".join(json.dumps({
        "kind": "evaluation", "source": "evaluations.csv", "data": row,
    }) for row in [
        {"slot": "1", "fitness": "-9", "status": "ok"},
        {"slot": "2", "fitness": "-7", "status": "ok"},
    ]) + "\n")
    monitor = ResultsMonitor(tmp_path, "traceaad_v9_19")
    row = monitor.overview("traceaad_v9_19")["tasks"][0]["runs"][0]
    assert row["budget_used"] == row["budget"] == 1000
    curve = monitor.run_detail("traceaad_v9_19", "tsp_construct", "rep1")["curve"]
    assert [{k: p[k] for k in ("evaluation", "value")} for p in curve] == [
        {"evaluation": 1, "value": 9.0}, {"evaluation": 2, "value": 7.0},
    ]

    write_json(run / "logs/run_summary.json", {"status": "unknown"})
    (run / "search.jsonl").write_text("\n".join(json.dumps({
        "kind": "node", "source": "nodes.jsonl", "data": node,
    }) for node in [
        {"id": 1, "fitness": -8.0, "code": "first"},
        {"id": 2, "fitness": -7.0, "code": "second"},
    ]) + "\n")
    detail = monitor.run_detail("traceaad_v9_19", "tsp_construct", "rep1")
    assert detail["status"] == "unknown"
    assert detail["best"]["code"] == "second"


def test_missing_final_summary_uses_recorded_evaluations(tmp_path):
    run = tmp_path / "traceaad_v9_7" / "tsp_construct" / "rep1"
    write_json(run / "run_config.json", {
        "task": "tsp_construct", "method_params": {"budget": 10},
    })
    write_json(run / "logs/run_summary.json", {"status": "unknown"})
    (run / "search.jsonl").write_text("\n".join(json.dumps({
        "kind": "candidate", "source": "artifacts/candidates.jsonl", "data": row,
    }) for row in [
        {"evaluator_called": True, "child_fitness": -9.0},
        {"evaluator_called": False, "child_fitness": None},
        {"evaluator_called": True, "child_fitness": -8.0},
    ]) + "\n")
    monitor = ResultsMonitor(tmp_path)
    row = monitor.overview("traceaad_v9_7")["tasks"][0]["runs"][0]
    assert (row["status"], row["budget_used"], row["valid_nodes"]) == ("unknown", 2, 2)


def append_records(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def candidate(index, fitness, **metadata):
    return {"kind": "candidate", "candidate_id": index, "budget_used": index,
            "fitness": fitness, "status": "ok" if fitness is not None else "evaluation_failed",
            "operator": "Refine", **metadata}


def test_breakthroughs_use_incumbent_not_parent_and_keep_operator_provenance(tmp_path):
    from experiments.infra.monitor_history import TrainingHistory
    records = [candidate(1, -12, operator="Init"),
               {"kind": "attempt", "data": {"id": 2, "scope": "Refine", "reference_mode": "Transfer",
                "channel": "trial", "parent_id": 1, "idea": "真实修改"}},
               candidate(2, -10), candidate(3, -11), candidate(4, -10),
               candidate(5, None), candidate(6, -9, operator="TRACE_RECHECK"), candidate(7, None)]
    append_records(tmp_path / "search.jsonl", records)
    points, recent, operators, outcomes = TrainingHistory(tmp_path, minimize=True).read()
    assert [p["evaluation"] for p in points] == [1, 2, 6, 7]
    assert [p["fitness"] for p in points] == [-12, -10, -9, -9]
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
    append_records(tmp_path / "search.jsonl", [candidate(i, i / 100) for i in range(1, 401)])
    points, _, _, _ = TrainingHistory(tmp_path, minimize=False).read()
    assert len(points) == 400
    assert [p["candidate"] for p in points] == list(range(1, 401))
    assert sum(p["kind"] == "breakthrough" for p in points) == 399
    assert all(p["fitness"] == p["value"] for p in points)


def test_live_history_retries_partial_utf8_tail_and_does_not_recount(tmp_path):
    from experiments.infra.monitor_history import TrainingHistory
    path = tmp_path / "search.jsonl"
    append_records(path, [candidate(1, 1)])
    reader = TrainingHistory(tmp_path, minimize=False)
    first = reader.read()
    offset = reader.offset
    payload = (json.dumps(candidate(2, 2, idea="改进"), ensure_ascii=False) + "\n").encode()
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
    path = tmp_path / "search.jsonl"
    append_records(path, [candidate(1, 1), candidate(2, 2)])
    reader = TrainingHistory(tmp_path, minimize=False)
    assert len(reader.read()[0]) == 2
    replacement = tmp_path / "replacement.jsonl"
    append_records(replacement, [candidate(1, 9)])
    replacement.replace(path)
    assert [p["fitness"] for p in reader.read()[0]] == [9]


def test_invalid_scores_never_create_breakthroughs(tmp_path):
    from experiments.infra.monitor_history import TrainingHistory
    append_records(tmp_path / "search.jsonl", [candidate(1, None), candidate(2, float("nan")),
                   candidate(3, float("inf")), candidate(4, 0), candidate(5, 0), candidate(6, -1)])
    points, _, _, _ = TrainingHistory(tmp_path, minimize=False).read()
    assert [(p["evaluation"], p["kind"]) for p in points] == [(4, "initial"), (6, "progress")]
    assert points[0]["fitness"] == 0


def test_overview_exposes_same_curve_as_detail_with_candidate_axis(tmp_path):
    run = tmp_path / "traceaad_v10_14" / "op_aco" / "rep1"
    write_json(run / "run_config.json", {"method": "v1014", "task": "op_aco", "repeat": 1})
    write_json(run / "logs/run_summary.json", {"status": "running", "budget": 1000,
               "best": {"fitness": 3, "code": "pass"}})
    append_records(run / "search.jsonl", [candidate(1, 1), candidate(2, 3)])
    monitor = ResultsMonitor(tmp_path)
    row = monitor.overview("traceaad_v10_14")["tasks"][0]["runs"][0]
    detail = monitor.run_detail("traceaad_v10_14", "op_aco", "rep1")
    assert row["curve"] == detail["curve"]
    assert row["x_label"] == detail["x_label"] == "候选尝试"
    assert row["curve"][-1]["gain"] == 2
