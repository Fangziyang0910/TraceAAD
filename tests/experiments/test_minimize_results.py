import hashlib
import json

import pytest

from experiments.infra.migrations.minimize_results import complete_live_run, convert_run, transform
from experiments.infra.monitor_history import TrainingHistory
from traceaad.common.selection import better, weights
from traceaad.common.storage import RESULT_FORMAT, append_jsonl, read_json, write_json


@pytest.mark.parametrize("task,old_fitness,old_score", [
    ("tsp_construct", -10., 10.), ("op_aco", 10., 10.), ("cob_tsp", .8, .8)])
def test_migration_preserves_winner_offsets_and_source(tmp_path, task, old_fitness, old_score):
    run = tmp_path / task / "rep1"
    run.mkdir(parents=True)
    write_json(run / "run_config.json", {"task": task, "result_format": "traceaad-results-v1",
                                         "objective": "min" if task == "tsp_construct" else "max"})
    code = "def solve(): return 1\n"
    key = hashlib.sha256(code.encode()).hexdigest()
    append_jsonl(run / "programs.jsonl", {"key": key, "code": code})
    append_jsonl(run / "calls.jsonl", {"prompt": "Higher is better. Score 10.", "response": code})
    program = {"id": 1, "key": key, "fitness": old_fitness, "score": old_score, "valid": True}
    append_jsonl(run / "events.jsonl", {"kind": "candidate", "candidate_id": 1,
        "budget_used": 1, "fitness": old_fitness, "program": program, "node_id": 1,
        "operator": "Init", "status": "valid", "valid": True, "attempt": {"id": 1},
        "evaluations": [{"score": old_fitness}], "ts": "2026-10-07T00:00:00+09:00"})
    boundary = (run / "events.jsonl").stat().st_size
    append_jsonl(run / "events.jsonl", {"kind": "progress", "ts": None})
    write_json(run / "resume.json", {"files": {"events.jsonl": boundary}, "state": {
        "selected_id": 1}})
    write_json(run / "summary.json", {"status": "finished", "best": program})
    write_json(run / "heldout.json", [{"task": task, "fitness": old_fitness,
        "scores": [old_fitness], "key": key, "node_id": 1}])
    untouched = {f: (run / f).read_bytes() for f in ("programs.jsonl", "calls.jsonl")}

    receipt = convert_run(run, tmp_path)
    assert receipt["frontiers"] == 1
    assert read_json(run / "run_config.json")["result_format"] == RESULT_FORMAT
    assert read_json(run / "summary.json")["best"] == {**program, "fitness": -old_fitness, "score": -old_fitness}
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    assert events[0]["evaluations"][0]["score"] == -old_fitness
    heldout = read_json(run / "heldout.json")[0]
    assert heldout["fitness"] == -old_fitness and heldout["scores"] == [-old_fitness]
    checkpoint = read_json(run / "resume.json")
    assert checkpoint["state"]["selected_id"] == 1
    assert checkpoint["files"]["events.jsonl"] == len((json.dumps(events[0], ensure_ascii=False, separators=(",", ":")) + "\n").encode())
    assert TrainingHistory(run, minimize=True).read()[0][0]["value"] == -old_fitness
    assert {f: (run / f).read_bytes() for f in untouched} == untouched
    assert convert_run(run, tmp_path) is None  # no double negation


def test_scalar_conversion_keeps_improvements_failures_and_physical_counts():
    row = {"task": "tsp_construct", "fitness": -5., "score": 5.,
           "gain": .2, "same_score": 10, "failure": {"seconds": 30}, "eval_score": -6.}
    assert transform(row, canonical=True) == {**row, "fitness": 5., "score": 5., "eval_score": 6.}
    assert transform({"op_aco": {"score": 10.}, "tsp_construct": {"score": 5.}}) == {
        "op_aco": {"score": -10.}, "tsp_construct": {"score": 5.}}


def test_lower_cost_is_better_and_parent_sampling_prefers_it():
    assert better(-10., -9.) and better(9., 10.)
    assert not better(10., 9.) and not better(9., 9.)
    nodes = [{"id": i, "fitness": float(i), "valid": True} for i in range(20)]
    probabilities, _ = weights(nodes, {}, {n["id"]: n for n in nodes})
    assert probabilities[0] > probabilities[-1]
    assert sum(probabilities) == pytest.approx(1.)


def test_running_legacy_writer_is_normalized_only_in_read_view(tmp_path):
    run = tmp_path / "cvrp_aco" / "rep1"
    write_json(run / "run_config.json", {"task": "cvrp_aco", "result_format": RESULT_FORMAT})
    write_json(run / ".minimize/live.json", {"task": "cvrp_aco"})
    for i, score in enumerate((10., -9.), 1):
        append_jsonl(run / "events.jsonl", {"kind": "candidate", "candidate_id": i,
            "budget_used": i, "fitness": score, "node_id": i, "operator": "Refine",
            "valid": True, "status": "valid", "attempt": {"id": i}})
    boundary = (run / "events.jsonl").stat().st_size
    write_json(run / "resume.json", {"files": {"events.jsonl": boundary}, "state": {"phase": "search"}})
    original = (run / "events.jsonl").read_bytes()
    points = TrainingHistory(run, minimize=True).read()[0]
    assert [p["fitness"] for p in points] == [10., 9.]
    assert points[-1]["gain"] == 1.
    assert (run / "events.jsonl").read_bytes() == original
    write_json(run / "summary.json", {"status": "finished", "best": {"fitness": -9., "score": 9.}})
    complete_live_run(run, tmp_path)
    assert not (run / ".minimize/live.json").exists()
    assert read_json(run / "summary.json")["best"] == {"fitness": 9., "score": 9.}
    assert read_json(run / "resume.json")["files"]["events.jsonl"] == (run / "events.jsonl").stat().st_size
