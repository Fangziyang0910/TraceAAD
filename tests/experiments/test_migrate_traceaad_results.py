import json

import pytest

from experiments.infra.artifacts import pick_best_sample
from experiments.infra.migrate_traceaad_results import migrate_run


def test_migrate_legacy_run_keeps_search_and_summary(tmp_path):
    run = tmp_path / "traceaad_v9_19" / "tsp_construct" / "rep1"
    (run / "checkpoints").mkdir(parents=True)
    (run / "logs").mkdir()
    (run / "run_config.json").write_text('{"task":"tsp_construct"}\n')
    (run / "logs" / "summary.json").write_text(json.dumps({
        "status": "finished", "best_score": 3.0, "best_algorithm_id": 2,
    }))
    (run / "best_program.py").write_text("def solve():\n    return 3\n")
    (run / "best_history.jsonl").write_text(json.dumps({
        "fitness": 3.0, "slot": 2, "program": "def solve():\n    return 3\n",
    }) + "\n")
    (run / "evaluations.csv").write_text("slot,fitness\n2,3.0\n")
    (run / "checkpoints" / "latest.json").write_text('{"next_slot":3}')

    assert migrate_run(run) == (4, 4)
    assert json.loads((run / "logs" / "run_summary.json").read_text())["status"] == "finished"
    assert not (run / "logs" / "summary.json").exists()
    assert not (run / "checkpoints").exists()
    records = [json.loads(line) for line in (run / "search.jsonl").read_text().splitlines()]
    assert {record["kind"] for record in records} == {"state", "best", "program", "evaluation"}
    assert next(record["data"] for record in records if record["kind"] == "state") == {"next_slot": 3}
    best, _ = pick_best_sample(run)
    assert (best["score"], best["sample_order"]) == (3.0, 2)
    assert migrate_run(run) == (0, 0)


def test_failed_conversion_leaves_originals(tmp_path):
    run = tmp_path / "rep1"
    run.mkdir()
    (run / "run_config.json").write_text("{}")
    (run / "events.jsonl").write_text("not JSON\n")

    with pytest.raises(json.JSONDecodeError):
        migrate_run(run)
    assert (run / "events.jsonl").exists()
    assert not (run / "search.jsonl").exists()


def test_best_history_remains_preferred_over_candidate_stream(tmp_path):
    run = tmp_path / "rep1"
    (run / "artifacts").mkdir(parents=True)
    (run / "logs").mkdir()
    (run / "run_config.json").write_text("{}")
    (run / "logs" / "summary.json").write_text('{"status":"finished"}')
    (run / "best_history.jsonl").write_text(json.dumps({
        "fitness": 2, "eval_count": 2, "program": "history",
    }) + "\n")
    (run / "artifacts" / "candidates.jsonl").write_text(json.dumps({
        "child_fitness": 3, "order": 3, "program": "candidate",
    }) + "\n")

    migrate_run(run)
    best, records = pick_best_sample(run)
    assert best["program"] == "history"
    assert len(records) == 1
