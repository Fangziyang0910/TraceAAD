import json

from core.training_monitor import V1013Monitor


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


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
    write_jsonl(run_dir / "events.jsonl", [
        {"candidate_id": 1, "budget_used": 1, "operator": "Init", "status": "ok", "fitness": -12.0},
        {"candidate_id": 2, "budget_used": 2, "operator": "Refine", "status": "ok", "fitness": -10.0},
        {"candidate_id": 3, "budget_used": 3, "operator": "Tune", "status": "eval_failed", "fitness": None},
    ])
    write_jsonl(run_dir / "nodes.jsonl", [
        {"id": 0, "fitness": -12.0, "operator": "Init", "idea": "first", "code": "def f(): return 1"},
        {"id": 1, "fitness": -10.0, "operator": "Refine", "idea": "best", "code": "def f(): return 2"},
    ])
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
