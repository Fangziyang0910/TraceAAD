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
    assert state["summary"] == {
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
    assert detail["curve"] == [
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
    assert monitor.run_detail("traceaad_v9_19", "tsp_construct", "rep1")["curve"] == [
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
