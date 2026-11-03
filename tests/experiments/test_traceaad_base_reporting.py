"""Offline contracts between independent Base and the existing result readers."""

from copy import deepcopy
import json
import time

import pytest

from benchmarks.tasks import PRIMARY_SPLITS, scale_of_split
from experiments.infra.artifacts import load_scored_samples, pick_best_sample
from experiments.infra.batch_status import collect, journal_tail, run_status
from experiments.infra.result_storage import Programs, save_heldout, selected_program
from experiments.infra.storage import read_json, rows, write_json
from experiments.infra.watch_server4 import complete
from experiments.traceaad_base.reporting import Reporter
from experiments.visualization.data import ResultsMonitor
from experiments.visualization.history import TrainingHistory
from traceaad_base import Config, ModelReply, Task, TraceAAD


class ScriptedModel:
    def __init__(self, replies):
        self.replies = iter(replies)

    def complete(self, prompt, **settings):
        reply = next(self.replies)
        if isinstance(reply, BaseException):
            raise reply
        return reply if isinstance(reply, ModelReply) else ModelReply(reply, usage={
            "prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10,
            "completion_tokens_details": {"reasoning_tokens": 2},
            "prompt_tokens_details": {"cached_tokens": 1},
        })


def candidate(value=3, *, body=None):
    return json.dumps({"analysis": "Use a constant score.", "design": "Return a constant score", "code":
                       "def solve(x):\n    " + (body or f"return {value}") + "\n"})


@pytest.fixture
def task():
    return Task("Return a finite score; lower is better.", "def solve(x): return x",
                lambda source, function: function(3), time_limit=3,
                example_code="def solve(x): return -100")


@pytest.fixture
def layout(tmp_path):
    directory = tmp_path / "traceaad_base" / "tsp_construct" / "offline_rep1"
    metadata = {
        "method": "traceaad_base", "experiment": "traceaad_base", "task": "tsp_construct",
        "run_name": directory.name, "batch": "offline", "seed": 0, "repeat": 1,
        "budget": 5, "budget_axis": "评价次数", "budget_basis": "evaluate_calls",
        "score_metric": "raw_objective", "llm": {"model": "offline-script"},
        "task_eval": {"time_limit": 3}, "custom_metadata": {"keep": True},
    }
    row = {"task": metadata["task"], "run_name": directory.name, "session": "offline"}
    manifest = directory.parent.parent / "batch_offline.json"
    write_json(manifest, {"batch": "offline", "budget_per_run": 5, "plan": [row]})
    return directory, metadata, manifest, row


def native_files(directory):
    return {p.relative_to(directory): p.read_bytes()
            for p in (directory / "native").rglob("*") if p.is_file()}


def publish_unchanged(reporter, search, result=None):
    before = native_files(reporter.directory)
    reporter.publish(search, result)
    assert native_files(reporter.directory) == before


def test_offline_search_readers_follow_evaluations_and_repaired_individuals(task, layout, tmp_path):
    directory, metadata, manifest, row = layout
    original_metadata = deepcopy(metadata)
    model = ScriptedModel([
        "not json", "{}",                         # No individual and no Evaluate.
        candidate(body="return 1 / 0"), candidate(8),  # Two evaluations, one individual.
        "not json", candidate(3),                  # Format repair uses one Evaluate.
        candidate(body="return float('nan')"), "{}",  # Preserve failed individual.
        candidate(3),                              # Equal code, distinct individual.
    ])
    search = TraceAAD(task, model, config=Config(evaluation_budget=5, initial_size=1),
                      run_dir=directory / "native")
    reporter = Reporter(directory, metadata)
    publish_unchanged(reporter, search)
    history, sources = TrainingHistory(directory), Programs(directory)
    monitor = ResultsMonitor(tmp_path, batch="offline")
    assert monitor.overview("traceaad_base")["summary"]["budget_used"] == 0
    assert history.progress_snapshot()["candidate_count"] == 0
    assert history.read()[0] == []
    assert sources.get(0) == task.example_code
    assert run_status(manifest.parent, row, 5, time.time())["status"] == "recorded"
    assert selected_program(directory) is None
    snapshots = []

    def on_attempt(attempt):
        publish_unchanged(reporter, search)
        progress = history.progress_snapshot()
        detail = monitor.run_detail("traceaad_base", "tsp_construct", directory.name)
        status = collect(manifest)["runs"][0]
        checkpoint, _, _ = journal_tail(directory / "events.jsonl")
        assert progress["x_label"] == detail["x_label"] == "评价次数"
        assert progress["budget_used"] == detail["budget_used"] == status["budget_used"]
        assert status["timing"]["unit"] == "评价"
        assert checkpoint["attempts"] == search.evaluations_used
        assert progress["candidate_count"] == attempt.id
        assert detail["timing"]["unit"] == "评价"
        assert selected_program(directory) is None
        if attempt.program_id is not None:
            assert sources.get(attempt.program_id) == search.archive[attempt.program_id].code
        snapshots.append((progress["budget_used"], progress["valid_nodes"], search.model_calls))

    result = search.run(on_attempt=on_attempt)
    assert snapshots == [(0, 0, 2), (2, 1, 4), (3, 2, 6), (4, 2, 8), (5, 3, 9)]
    publish_unchanged(reporter, search, result)
    summary = read_json(directory / "summary.json")
    assert read_json(directory / "run_config.json") == original_metadata == metadata
    assert summary["status"] == "finished"
    assert summary["stop_reason"] == "evaluation_budget_exhausted"
    assert summary["budget_used"] == summary["evaluation_calls"] == summary["budget"] == 5
    assert summary["model_calls"] == 9
    assert summary["attempts"] == summary["candidate_count"] == 5
    assert summary["program_count"] == 4 and summary["num_nodes"] == 3
    assert summary["failed_programs"] == 1 and summary["repair_attempts"] == 4
    assert summary["format_failures"] == 4
    assert summary["evaluation_outcomes"] == {"error": 2, "success": 3}
    assert summary["best"]["analysis"] == search.best.analysis == "Use a constant score."
    usage = summary["token_usage"]
    assert (usage["calls"], usage["calls_with_usage"], usage["calls_without_usage"]) == (9, 9, 0)
    assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (
        63, 27, 90)
    assert (usage["reasoning_tokens"], usage["cached_prompt_tokens"]) == (18, 9)

    events = [r for r in rows(directory / "events.jsonl") if r["kind"] != "progress"]
    events = [r for r in events if r["kind"] in {"candidate", "delivery_failed"}]
    assert [r["node_id"] for r in events] == [None, 1, 2, 3, 4]
    assert [r["budget_used"] for r in events] == [0, 2, 3, 4, 5]
    assert [r["status"] for r in events] == [
        "invalid_output", "valid", "valid", "eval_failed", "valid"]
    assert events[0]["attempt"]["reason"]
    assert len(events[1]["attempt"]["evaluations"]) == 2
    assert events[1]["program"]["parent_id"] == 0
    assert events[1]["operator"] == "Init"
    assert events[2]["program"]["reference_id"] == search.archive[2].reference_id
    assert events[3]["program"]["evaluation"]["status"] == "error"
    assert "non-finite fitness" in events[3]["attempt"]["reason"]
    assert events[3]["attempt"]["calls"][-1]["format_failure"]
    assert [r["id"] for r in rows(directory / "programs.jsonl")] == [0, 1, 2, 3, 4]
    curve, _, _, outcomes = history.read()
    assert [(p["evaluation"], p["fitness"]) for p in curve] == [(2, 8), (3, 3), (5, 3)]
    assert outcomes == {"invalid_output": 1, "valid": 3, "eval_failed": 1}
    assert history.node(3)["id"] == 2  # Attempt 3 produced native individual 2.
    assert history.node(1, by_node=True)["code"] == search.archive[1].code
    chosen, samples = pick_best_sample(directory)
    assert (chosen["node_id"], chosen["sample_order"], chosen["score"]) == (2, 3, 3)
    assert [s["node_id"] for s in samples] == [1, 2, 4]
    assert len(load_scored_samples(directory, max_sample_order=2)) == 1
    assert chosen["program"] == selected_program(directory)["code"] == result.best.code
    assert selected_program(directory)["verified"]
    assert read_json(directory / "selection.json")["selected_node"] == result.best.id
    detail = monitor.run_detail("traceaad_base", "tsp_construct", directory.name)
    assert detail["status"] == "finished" and detail["best_fitness"] == 3
    assert detail["selected_best"]["id"] == 2
    assert detail["selected_best"]["code"] == result.best.code
    assert collect(manifest)["runs"][0]["frozen"]
    assert not complete(directory)  # The watcher still requires heldout results.

    # Exercise only the heldout storage contract, without evaluating any heldout task.
    for split in PRIMARY_SPLITS["tsp_construct"]:
        save_heldout(directory, {"task": "tsp_construct", "node_id": 2,
                                "verification": "verified", "fitness": 4.0,
                                "scale": str(scale_of_split("tsp_construct", split))})
    assert complete(directory)
    assert collect(manifest)["runs"][0]["heldout"]["valid"] == list(PRIMARY_SPLITS["tsp_construct"])
    compared = monitor.compare(["traceaad_base"])["tasks"][0]["cohorts"][0]["runs"][0]
    assert compared["train"] == 3 and set(compared["heldout"].values()) == {4.0}

    before = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    publish_unchanged(Reporter(directory, metadata), search, result)
    assert {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()} == before
    assert history.progress_snapshot()["candidate_count"] == 5


def test_resume_rebuilds_deleted_or_corrupt_derived_logs(task, layout):
    directory, metadata, _, _ = layout
    config = Config(evaluation_budget=5, initial_size=1)
    with TraceAAD(task, ScriptedModel([candidate(body="return 1 / 0"), candidate(8)]),
                  config=config, run_dir=directory / "native") as search:
        reporter = Reporter(directory, metadata)
        search.step()
        publish_unchanged(reporter, search)
    history, sources = TrainingHistory(directory), Programs(directory)
    assert history.progress_snapshot()["budget_used"] == 2
    assert sources.get(1) == search.archive[1].code
    for name in ("run_config.json", "events.jsonl", "programs.jsonl", "summary.json"):
        (directory / name).write_text("broken partial artifact", encoding="utf-8")
    (directory / "selection.json").write_text('{"selected_node": 999}', encoding="utf-8")
    (directory / "best_program.py").write_text("stale code", encoding="utf-8")
    metadata["budget_axis"], metadata["budget_basis"] = "生成次数", "model_calls"
    resumed = TraceAAD(task, ScriptedModel([candidate(2)] * 3),
                       run_dir=directory / "native", resume=True)
    reporter = Reporter(directory, metadata)
    publish_unchanged(reporter, resumed)
    assert history.progress_snapshot()["candidate_count"] == 1
    assert history.progress_snapshot()["budget_used"] == 2
    assert sources.get(1) == resumed.archive[1].code
    config_view = read_json(directory / "run_config.json")
    assert config_view["budget_axis"] == "评价次数" and config_view["budget_basis"] == "evaluate_calls"
    assert not (directory / "best_program.py").exists()
    assert not (directory / "selection.json").exists()
    result = resumed.run(on_attempt=lambda _: reporter.publish(resumed))
    publish_unchanged(reporter, resumed, result)
    assert result.model_calls == result.evaluations == 5
    assert history.progress_snapshot()["candidate_count"] == 4
    assert selected_program(directory)["id"] == 2


@pytest.mark.parametrize("format_failure", [False, True])
def test_unevaluated_limit_is_terminal_without_consuming_budget(task, layout, format_failure):
    directory, metadata, manifest, _ = layout
    metadata["budget"] = 1000
    replies = ["{}"] * 20 if format_failure else [RuntimeError("offline failure")] * 10
    search = TraceAAD(task, ScriptedModel(replies), config=Config(evaluation_budget=1000),
                      run_dir=directory / "native")
    reporter = Reporter(directory, metadata)
    reporter.publish(search)
    result = search.run(on_attempt=lambda _: reporter.publish(search))
    publish_unchanged(reporter, search, result)
    summary = read_json(directory / "summary.json")
    assert summary["status"] == "no_valid_root" and summary["native_status"] == "failed"
    assert summary["stop_reason"] == "consecutive_unevaluated_limit"
    assert summary["budget_used"] == summary["num_nodes"] == summary["program_count"] == 0
    assert summary["attempts"] == 10
    assert summary["model_calls"] == (20 if format_failure else 10)
    assert summary["token_usage"]["calls_without_usage"] == (0 if format_failure else 10)
    history = TrainingHistory(directory)
    assert history.progress_snapshot()["candidate_count"] == 10
    assert history.read()[3] == {"invalid_output" if format_failure else "delivery_failed": 10}
    assert collect(manifest)["runs"][0]["budget_used"] == 0
    assert read_json(directory / "selection.json")["selected_node"] is None
    assert selected_program(directory) is None and not (directory / "best_program.py").exists()
    assert complete(directory)


def test_interrupted_evaluation_still_consumes_budget_after_native_resume(task, layout):
    directory, metadata, _, _ = layout
    metadata["budget"] = 2
    with TraceAAD(task, ScriptedModel([candidate()]), config=Config(evaluation_budget=2),
                  run_dir=directory / "native") as first:
        def interrupt(code):
            raise KeyboardInterrupt()

        first.evaluator.evaluate = interrupt
        with pytest.raises(KeyboardInterrupt):
            first.step()
    search = TraceAAD(task, ScriptedModel([ModelReply(candidate(), usage={"total_tokens": 11})]),
                      run_dir=directory / "native", resume=True)
    reporter = Reporter(directory, metadata)
    publish_unchanged(reporter, search)
    history = TrainingHistory(directory)
    assert history.progress_snapshot()["budget_used"] == 1
    assert history.progress_snapshot()["valid_nodes"] == 0
    interrupted = next(r for r in rows(directory / "events.jsonl")
                       if r["kind"] == "delivery_failed")
    assert interrupted["node_id"] is None and interrupted["attempt"]["interruption"]
    result = search.run()
    publish_unchanged(reporter, search, result)
    assert history.progress_snapshot()["budget_used"] == 2
    assert history.progress_snapshot()["candidate_count"] == 2
    assert history.progress_snapshot()["valid_nodes"] == 1
    assert selected_program(directory)["id"] == 1
    assert read_json(directory / "summary.json")["token_usage"]["total_tokens"] == 21


@pytest.mark.parametrize("body,status", [("return 1 / 0", "eval_failed"),
                                         ("while True: pass", "timeout")])
def test_failed_last_evaluate_is_one_failed_individual(task, layout, body, status):
    directory, metadata, manifest, _ = layout
    metadata["budget"] = 1
    if status == "timeout":
        task.time_limit = .1
    search = TraceAAD(task, ScriptedModel([candidate(body=body)]),
                      config=Config(evaluation_budget=1), run_dir=directory / "native")
    reporter = Reporter(directory, metadata)
    reporter.publish(search)
    result = search.run(on_attempt=lambda _: reporter.publish(search))
    publish_unchanged(reporter, search, result)
    summary = read_json(directory / "summary.json")
    assert summary["status"] == "no_valid_root"
    assert summary["stop_reason"] == "evaluation_budget_exhausted"
    assert summary["budget_used"] == summary["model_calls"] == summary["failed_programs"] == 1
    assert summary["repair_attempts"] == summary["num_nodes"] == 0
    history = TrainingHistory(directory)
    assert history.read()[3] == {status: 1}
    assert history.node(1)["reason"]
    assert collect(manifest)["runs"][0]["budget_used"] == 1
    assert complete(directory) and selected_program(directory) is None


def test_early_stop_exports_the_best_without_claiming_a_full_budget(task, layout):
    directory, metadata, manifest, _ = layout
    search = TraceAAD(task, ScriptedModel([candidate(), RuntimeError("offline failure")]),
                      config=Config(evaluation_budget=5, max_unevaluated=1),
                      run_dir=directory / "native")
    reporter = Reporter(directory, metadata)
    reporter.publish(search)
    result = search.run(on_attempt=lambda _: reporter.publish(search))
    publish_unchanged(reporter, search, result)
    summary = read_json(directory / "summary.json")
    assert summary["status"] == "finished" and summary["stop_reason"] == "consecutive_unevaluated_limit"
    assert summary["budget_used"] == 1 and summary["budget"] == 5
    assert summary["attempts"] == summary["model_calls"] == 2
    assert selected_program(directory)["id"] == 1
    assert collect(manifest)["runs"][0]["budget_used"] == 1
