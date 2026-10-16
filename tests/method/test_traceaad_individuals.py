"""The individual is the parsed program, independent of code equality."""

import json

import pytest

from tests.support import TinyEvaluation, TokenLLM, response
from traceaad.v12_1 import Config, Program, TraceAADV121
from traceaad.v12_1.state import Facts
from traceaad.v12_1.storage import append_jsonl, rows


def search(directory, *answers, **options):
    return TraceAADV121(
        evaluation=TinyEvaluation(),
        llm=TokenLLM(*answers),
        run_dir=directory,
        config=Config(**{"budget": 3, "roots": 1, "linking": False, **options}),
    )


def test_parse_failure_has_no_individual_and_does_not_shift_individual_identity(
    tmp_path,
):
    first = search(tmp_path, "{bad JSON", response(1))
    first._roots()
    assert first.programs == {} and first.budget_used == 1
    resumed = search(tmp_path, response(1))
    resumed._roots()
    individual = resumed.programs[1]
    assert isinstance(individual, Program) and individual.generation == 2
    assert individual.id == 1 and resumed.budget_used == 2
    assert not hasattr(resumed, "attempts_table") and not hasattr(
        resumed.facts, "attempts"
    )
    events = list(rows(tmp_path / "events.jsonl"))
    assert events[0]["kind"] == "delivery_failed" and "program" not in events[0]
    assert events[1]["program"]["id"] == 1 and all("attempt" not in e for e in events)


def test_identical_code_has_independent_analysis_sources_and_evaluations(tmp_path):
    replies = [
        json.dumps({"analysis": why, "design": why, "code": "def score(x): return 1"})
        for why in ("first", "second")
    ]
    method = search(tmp_path, *replies)
    method._roots()
    parent = method.programs[1]
    child = method._create_program(
        method.prompts.build("Refine", parent), parent=parent
    )
    assert parent is method.programs[1] and child is method.programs[2]
    assert (
        parent.code == child.code
        and parent.analysis == "first"
        and child.analysis == "second"
    )
    assert child.parent_id == parent.id and len(method.facts.evaluations) == 2
    sources = list(rows(tmp_path / "programs.jsonl"))
    assert [p["id"] for p in sources] == [1, 2]
    restored = search(tmp_path)
    assert [p.analysis for p in restored.programs.values()] == ["first", "second"]
    assert restored.prompts.build("Refine", restored.programs[2])[
        "history_edge_ids"
    ] == [2]


def test_identical_failing_code_is_evaluated_again_for_repair(tmp_path):
    method = search(tmp_path, response("1 / 0"), response("1 / 0"))
    method._roots()
    assert len(method.programs) == len(method.facts.evaluations) == 2
    first, repaired = method.programs.values()
    assert (
        first.status == repaired.status == "runtime_error"
        and repaired.repair_of == first.id
    )
    assert first.code == repaired.code and first.analysis == repaired.analysis == ""


def test_uncommitted_tail_is_excluded_then_replaced_on_resume(tmp_path):
    method = search(tmp_path, response(1))
    method._roots()
    append_jsonl(tmp_path / "events.jsonl", {"kind": "uncommitted"})
    append_jsonl(tmp_path / "programs.jsonl", {"key": "uncommitted", "code": "bad"})
    resumed = search(tmp_path, response(2))
    assert list(resumed.programs) == [1]
    parent = resumed.programs[1]
    resumed._create_program(resumed.prompts.build("Refine", parent), parent=parent)
    assert all(e["kind"] != "uncommitted" for e in rows(tmp_path / "events.jsonl"))
    assert [p["id"] for p in rows(tmp_path / "programs.jsonl")] == [1, 2]


def test_old_attempt_format_is_rejected_without_changing_files(tmp_path):
    append_jsonl(tmp_path / "events.jsonl", {"kind": "candidate", "attempt": {"id": 1}})
    before = (tmp_path / "events.jsonl").read_bytes()
    with pytest.raises(ValueError, match="old attempt/program format"):
        Facts(tmp_path)
    assert (tmp_path / "events.jsonl").read_bytes() == before


def test_monitor_reads_parse_failures_and_program_analysis(tmp_path):
    from experiments.infra.monitor_history import TrainingHistory

    method = search(tmp_path, "{bad", response(1))
    method._roots()
    method._roots()
    history = TrainingHistory(tmp_path, minimize=True)
    history.read()
    progress = history.progress_snapshot()
    assert progress["budget_used"] == progress["candidate_count"] == 2
    assert progress["valid_candidate_count"] == 1
    node = history.node(2)
    assert (
        node["id"] == 1
        and node["analysis"] == ""
        and node["code"] == method.programs[1].code
    )


@pytest.mark.parametrize("old_format", [False, True])
def test_cli_checks_saved_format_and_configuration_before_writing(
    tmp_path, monkeypatch, old_format
):
    from experiments.infra import search_run
    from experiments.traceaad_v12_1.run import main
    from traceaad.v12_1.storage import read_json, write_json

    directory = tmp_path / "batch" / "tsp_construct" / "saved"
    method = search(directory, response(1))
    method._roots()
    write_json(
        directory / "run_config.json", {"sentinel": "keep this original configuration"}
    )
    if old_format:
        checkpoint = read_json(directory / "resume.json")
        del checkpoint["state"]["individual_format"]
        write_json(directory / "resume.json", checkpoint)
    before = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    monkeypatch.setattr(search_run, "RESULTS_ROOT", tmp_path)
    with pytest.raises(
        ValueError, match="old attempt/program format|same configuration"
    ):
        main(
            ["--task", "tsp_construct", "--experiment", "batch", "--run-name", "saved"]
        )
    assert {
        p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()
    } == before


def test_monitor_updates_a_pending_individual_after_resuming_evaluation(
    tmp_path, monkeypatch
):
    from core.scheduling import SchedulerError
    from experiments.infra.monitor_history import TrainingHistory

    method = search(tmp_path, response(1), budget=1)

    def offline(*args, **kwargs):
        raise SchedulerError("offline")

    monkeypatch.setattr(method.training, "evaluate", offline)
    assert method.run()["status"] == "service_unavailable"
    history = TrainingHistory(tmp_path, minimize=True)
    history.read()
    assert history.progress_snapshot()["valid_candidate_count"] == 0
    resumed = search(tmp_path, budget=1)
    assert resumed.run()["status"] == "finished" and not resumed.llm.calls
    assert history.progress_snapshot()["candidate_count"] == 1
    assert history.progress_snapshot()["valid_candidate_count"] == 1
    assert history.node(1)["status"] == "valid"
