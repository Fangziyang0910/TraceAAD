"""Program identity, generation budget, repair, selection and recovery."""

import json
import random

import pytest

from core import llm as core_llm
from core.scheduling import SchedulerError
from tests.support import TinyEvaluation, TokenLLM, response
from traceaad import Config, TraceAAD
from traceaad.delivery import DeliveryError, optional_field, parse_json_response
from traceaad.selection import choose_reference, experience, temperature, weights
from traceaad.storage import append_jsonl, rows


def search(directory, *answers, **options):
    return TraceAAD(
        evaluation=TinyEvaluation(),
        llm=TokenLLM(*answers),
        run_dir=directory,
        config=Config(**{"budget": 6, "roots": 1, **options}),
    )


def test_generation_budget_and_training_winner(tmp_path):
    method = search(tmp_path, *(response(i) for i in range(6)))
    summary = method.run()
    assert summary["status"] == "finished"
    assert summary["budget_used"] == summary["model_calls"] == len(method.programs) == 6
    assert summary["best"]["fitness"] == -5
    assert summary["selection_evaluations"] == 0
    assert all(
        params["response_format"] == {"type": "json_object"} for _, params in method.llm.calls
    )
    assert method.evaluation_calls == 6


def test_parse_failure_does_not_create_an_individual(tmp_path):
    method = search(tmp_path, "{invalid JSON", response(1))
    method._roots()
    assert method.budget_used == 1 and not method.programs
    method._roots()
    assert list(method.programs) == [1]
    assert method.programs[1].generation == 2
    events = list(rows(tmp_path / "events.jsonl"))
    assert events[0]["kind"] == "delivery_failed" and "program" not in events[0]


def test_a_reply_enclosed_in_one_json_fence_is_accepted(tmp_path):
    reply = json.dumps({"analysis": "", "design": "return 1", "code": "def score(x):\n    return 1\n", "change": "c"})
    fenced = "```json\n" + reply + "\n```"
    assert parse_json_response(fenced, "stop") == parse_json_response(reply, "stop")
    assert optional_field(fenced, "change") == "c"
    with pytest.raises(DeliveryError):
        parse_json_response("Here it is:\n" + fenced, "stop")
    method = search(tmp_path, fenced)
    method._roots()
    assert list(method.programs) == [1] and method.programs[1].fitness == -1


def test_same_code_keeps_separate_analysis_and_evaluations(tmp_path):
    method = search(tmp_path, response(1, analysis="first"), response(1, analysis="second"))
    method._roots()
    parent = method.programs[1]
    method._create_program(method.prompts.build("Refine", parent), parent=parent)
    assert method.programs[1].code == method.programs[2].code
    assert [p.analysis for p in method.programs.values()] == ["first", "second"]
    assert method.programs[2].parent_id == 1 and method.evaluation_calls == 2
    sources = list(rows(tmp_path / "programs.jsonl"))
    assert [p["id"] for p in sources] == [1, 2]
    restored = search(tmp_path)
    assert [p.analysis for p in restored.programs.values()] == ["first", "second"]
    assert [p.code for p in restored.programs.values()] == [p["code"] for p in sources]


@pytest.mark.parametrize("bad", ["(", "1 / 0"])
def test_failed_program_is_repaired_once_then_initialization_continues(tmp_path, bad):
    method = search(
        tmp_path,
        response(bad, analysis="failure"),
        response(bad, analysis="repair"),
        response(7),
        budget=1,
    )
    summary = method.run()
    assert summary["status"] == "finished" and summary["budget_used"] == 3
    first, repaired, new_root = method.programs.values()
    assert not first.valid and not repaired.valid and new_root.valid
    assert repaired.repair_of == first.id and repaired.idea == first.idea
    assert repaired.analysis == "repair" and new_root.parent_id is None
    assert [p.action for p in method.programs.values()] == ["Init", "Repair", "Init"]


def test_initialization_stops_without_a_valid_program(tmp_path):
    method = search(tmp_path, *[response("1 / 0")] * 4, init_attempt_limit=4)
    summary = method.run()
    assert summary["status"] == "no_valid_root" and summary["budget_used"] == 4
    assert not (tmp_path / "best_program.py").exists()


def test_initialization_finishes_its_roots_before_applying_search_budget(tmp_path):
    method = search(tmp_path, response(1), response(2), response(3), roots=3, budget=1)
    summary = method.run()
    assert summary["num_roots"] == summary["budget_used"] == 3


def test_resume_keeps_choices_prompts_and_winner(tmp_path):
    answers = [response(i) for i in range(8)]
    control = search(tmp_path / "control", *answers, budget=8, roots=2)
    expected = control.run()
    first = search(tmp_path / "resumed", *answers[:4], budget=8, roots=2)
    first._roots()
    first._roots()
    first._roots()
    first._search()
    first._search()
    resumed = search(tmp_path / "resumed", *answers[4:], budget=8, roots=2)
    actual = resumed.run()
    assert first.llm.calls + resumed.llm.calls == control.llm.calls
    assert actual["best"]["id"] == expected["best"]["id"]
    assert actual["model_calls"] == expected["model_calls"] == 8


def test_interrupted_evaluation_resumes_without_regeneration(tmp_path, monkeypatch):
    method = search(tmp_path, response(1), budget=1)

    def unavailable(*args):
        raise SchedulerError("offline")

    monkeypatch.setattr(method.training, "evaluate", unavailable)
    assert method.run()["status"] == "service_unavailable"
    assert method.programs[1].status == "pending"
    resumed = search(tmp_path, budget=1)
    assert resumed.run()["status"] == "finished" and not resumed.llm.calls
    assert resumed.budget_used == resumed.evaluation_calls == 1


def test_uncommitted_journal_tail_is_not_restored(tmp_path):
    method = search(tmp_path, response(1))
    method._roots()
    append_jsonl(tmp_path / "events.jsonl", {"kind": "uncommitted"})
    append_jsonl(tmp_path / "programs.jsonl", {"id": 999, "code": "uncommitted"})
    resumed = search(tmp_path, response(2))
    assert list(resumed.programs) == [1]
    resumed._search("Refine")
    assert all(e["kind"] != "uncommitted" for e in rows(tmp_path / "events.jsonl"))
    assert [p["id"] for p in rows(tmp_path / "programs.jsonl")] == [1, 2]


def test_model_failure_records_cost_without_spending_generation_budget(tmp_path, monkeypatch):
    method = search(tmp_path)
    monkeypatch.setattr(core_llm, "SERVICE_WAIT_SECONDS", 0)

    def unavailable(*args, **kwargs):
        raise ConnectionError("offline")

    monkeypatch.setattr(method.llm, "draw_sample_with_details", unavailable)
    assert method.run()["status"] == "service_unavailable"
    assert method.progress.model_calls == 1 and method.budget_used == 0
    assert len(list(rows(tmp_path / "calls.jsonl"))) == 1


def test_parent_experience_credits_repair_to_original_child(tmp_path):
    method = search(tmp_path, response(1), response("1/0"), response(2))
    method._roots()
    method._search("Explore")
    tried, improved = experience(method.programs)
    assert tried[1] == improved[1] == 1
    probabilities, details = weights(list(method.archive.values()), method.programs)
    assert sum(probabilities) == pytest.approx(1)
    assert all(p > 0 for p in probabilities)
    beta, ess = temperature(range(20))
    assert beta > 0 and ess == pytest.approx(8)


def test_crossover_reference_uses_quality_and_excludes_parent(tmp_path):
    method = search(tmp_path, response(1), response(2), response(3), roots=3)
    method._roots()
    method._roots()
    method._roots()
    reference, _ = choose_reference(method.programs[3], method.archive, random.Random(0))
    assert reference.id == 2
    assert (
        choose_reference(method.programs[3], {3: method.programs[3]}, random.Random(0))[0] is None
    )
