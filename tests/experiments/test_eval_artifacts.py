import json

import pytest

from experiments.infra.artifacts import pick_best_sample
from tests.experiments.test_training_monitor import append_records, candidate, write_json


def test_pick_best_sample_requires_finished_by_default(tmp_path):
    run = tmp_path / "partial_run"
    write_json(run / "summary.json", {"status": "error", "budget_used": 50})
    append_records(run / "events.jsonl", [candidate(7, 1.25, code="def solve():\n    return 1\n")])
    with pytest.raises(RuntimeError, match="not a completed search"):
        pick_best_sample(run)
    best, records = pick_best_sample(run, allow_incomplete=True)
    assert best["score"] == 1.25 and best["sample_order"] == 7 and len(records) == 1


def test_pick_best_sample_joins_compact_node_to_its_event(tmp_path):
    run = tmp_path / "compact_traceaad"
    write_json(run / "run_config.json", {"task": "tsp_construct"})
    row = candidate(9, 2.5, code="def score(x):\n    return x", node_id=3)
    row["program"]["id"] = 3
    row["attempt"]["program_id"] = 3
    append_records(run / "events.jsonl", [row])
    best_meta = json.loads((run / "events.jsonl").read_text())["program"]
    write_json(run / "summary.json", {"status": "finished", "best": best_meta})
    best, records = pick_best_sample(run)
    assert best["sample_order"] == 9
    assert best["node_id"] == records[0]["node_id"] == 3
    assert best["key"] == records[0]["key"]
    assert best["program"] == "def score(x):\n    return x"
