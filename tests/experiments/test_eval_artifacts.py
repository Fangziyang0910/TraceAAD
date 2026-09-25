import json

import pytest

from experiments.infra.artifacts import pick_best_sample


def test_pick_best_sample_requires_finished_by_default(tmp_path):
    run_dir = tmp_path / "partial_run"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "logs" / "summary.json").write_text(
        json.dumps(
            {
                "status": "error",
                "budget_slots": 50,
                "best_score": 1.25,
                "best_algorithm_id": 7,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "best_program.py").write_text("def solve():\n    return 1\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="not a completed search"):
        pick_best_sample(run_dir)

    best, records = pick_best_sample(run_dir, allow_incomplete=True)

    assert best["score"] == 1.25
    assert best["sample_order"] == 7
    assert len(records) == 1


def test_pick_best_sample_joins_compact_node_to_its_event(tmp_path):
    run_dir = tmp_path / "compact_traceaad"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "logs" / "run_summary.json").write_text(json.dumps({
        "status": "finished",
        "best": {
            "id": 3,
            "code": "def score(x):\n    return x",
            "fitness": 2.5,
            "operator": "Refine",
        },
    }))
    (run_dir / "events.jsonl").write_text(
        json.dumps({"node_id": 3, "evaluation_id": 9}) + "\n",
    )

    best, records = pick_best_sample(run_dir)

    assert best["sample_order"] == 9
    assert records == [best]
