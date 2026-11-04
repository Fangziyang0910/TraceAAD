import json

import pytest

from traceaad_base import Config, TraceAAD


def test_format_then_program_error_share_one_repair(task, reply, scripted_model):
    model = scripted_model(["not json", reply(body="raise ValueError('bad calculation')"), reply(2)])
    search = TraceAAD(task, model, config=Config(evaluation_budget=2, initial_size=1))
    result = search.run()
    assert (result.evaluations, result.model_calls, result.attempts) == (2, 3, 2)
    assert search.archive[1].evaluation.status == "error"
    assert search.archive[2].valid
    assert [p.operator for p in search.archive.values()] == ["Init", "Init"]
    assert search.attempts[0].repair_used and len(search.attempts[0].evaluations) == 1
    assert "[Parsing Feedback]" in model.prompts[1]


def test_repair_replaces_same_individual_and_keeps_both_evaluated_versions(
    task, reply, scripted_model, tmp_path
):
    model = scripted_model([reply(body="raise ValueError('broken')", design="Original idea",
                                  analysis="Original computational change"),
                            reply(4, design="Repaired idea", analysis="Repaired computational change"), reply(2)])
    search = TraceAAD(task, model, config=Config(evaluation_budget=3, initial_size=1), run_dir=tmp_path)
    result = search.run()
    assert (result.evaluations, result.model_calls, result.valid_programs) == (3, 3, 2)
    first = search.archive[1]
    assert (first.parent_id, first.operator, first.reference_id) == (0, "Init", None)
    assert (first.design, first.evaluation.fitness) == ("Repaired idea", 4)
    assert first.analysis == "Repaired computational change"
    rows = [json.loads(line) for line in (tmp_path / "attempts.jsonl").read_text().splitlines()]
    assert rows[0]["program"]["analysis"] == first.analysis
    assert search.archive[2].parent_id == 1
    assert search.archive[2].operator in {"Refine", "Explore"}
    evaluations = [e for row in rows for e in row["attempt"]["evaluations"]]
    assert [e["program_id"] for e in evaluations] == [1, 1, 2]
    assert [e["status"] for e in evaluations] == ["error", "success", "success"]
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert json.loads(calls[0]["response"])["design"] == "Original idea"
    assert json.loads(calls[1]["response"])["design"] == "Repaired idea"
    assert "Analysis: Original computational change" in model.prompts[1]
    assert "Analysis: Repaired computational change" in model.prompts[2]
    assert "ValueError: broken" in model.prompts[1]
    assert "[Failure Feedback]" in model.prompts[1]
    assert (tmp_path / "best.py").read_text() == result.best.code


@pytest.mark.parametrize("analysis", [None, "", "  ", 3])
def test_invalid_analysis_is_repaired_before_evaluation(task, reply, scripted_model, analysis):
    raw = {"design": "Return a constant", "code": "def solve(x): return 1"}
    if analysis is not None:
        raw["analysis"] = analysis
    model = scripted_model([json.dumps(raw), reply(2, analysis="Use a constant score.")])
    search = TraceAAD(task, model, config=Config(evaluation_budget=1, initial_size=1))
    result = search.run()
    assert (result.evaluations, result.model_calls, result.attempts) == (1, 2, 1)
    assert search.attempts[0].repair_used
    assert result.best.analysis == "Use a constant score."
    assert "analysis" in search.attempts[0].calls[0]["format_failure"]


@pytest.mark.parametrize("failed_repair", ["{}", RuntimeError("model unavailable")])
def test_unusable_repair_preserves_existing_failure(task, reply, scripted_model, failed_repair):
    model = scripted_model([reply(body="raise RuntimeError('original')"), failed_repair])
    with TraceAAD(task, model, config=Config(evaluation_budget=2)) as search:
        attempt = search.step()
        assert (search.evaluations_used, search.model_calls) == (1, 2)
        assert len(search.archive) == 1
        assert attempt.program_id == 1 and attempt.repair_used
        assert "RuntimeError: original" in search.archive[1].evaluation.failure
        assert "original" in search.archive[1].code


def test_five_valid_initial_individuals_no_dedup_and_example_is_not_a_candidate(
    task, reply, scripted_model
):
    task.example_code = "def solve(x):\n    return -100\n"
    model = scripted_model([reply(2)] * 6)
    search = TraceAAD(task, model, config=Config(evaluation_budget=6))
    result = search.run()
    assert list(search.archive) == list(range(7))
    assert search.archive[0].evaluation is None and not search.archive[0].valid
    assert [p.operator for p in list(search.archive.values())[1:6]] == ["Init"] * 5
    assert search.archive[6].operator in {"Refine", "Explore", "Crossover"}
    assert result.best.id == 1 and result.best.evaluation.fitness == 2
    assert result.valid_programs == 6 and result.evaluations == 6
    assert all(s["temperature"] == 1.0 and s["max_tokens"] == 16384 for s in model.settings)


def test_last_evaluation_cannot_start_a_repair(task, reply, scripted_model):
    model = scripted_model([reply(body="return float('nan')")])
    search = TraceAAD(task, model, config=Config(evaluation_budget=1))
    result = search.run()
    assert (result.status, result.evaluations, result.model_calls) == ("failed", 1, 1)
    assert result.stop_reason == "evaluation_budget_exhausted"
    assert not search.attempts[0].repair_used and 0 not in search.archive


@pytest.mark.parametrize("responses,call_count", [([RuntimeError("offline")] * 10, 10), (["{}"] * 20, 20)])
def test_ten_unevaluated_attempts_stop_without_spending_evaluation_budget(
    task, scripted_model, responses, call_count
):
    search = TraceAAD(task, scripted_model(responses))
    result = search.run()
    assert (result.status, result.attempts, result.evaluations) == ("failed", 10, 0)
    assert result.model_calls == call_count and search.archive == {}
    assert result.stop_reason == "consecutive_unevaluated_limit"


def test_any_evaluation_resets_unevaluated_counter_and_failed_repair_is_one_child(
    task, reply, scripted_model
):
    model = scripted_model([RuntimeError("offline"), "{}", "{}",
                            reply(body="return 1 / 0"), reply(body="return 1 / 0")])
    with TraceAAD(task, model, config=Config(evaluation_budget=2)) as search:
        search.step()
        search.step()
        assert search.consecutive_unevaluated == 2
        attempt = search.step()
        assert search.consecutive_unevaluated == 0
        assert len(search.archive) == 1 and search.archive[1].evaluation.status == "error"
        assert len(attempt.evaluations) == 2 and attempt.model_calls == 2


def test_resume_at_attempt_boundary_reproduces_selection_and_archive(task, reply, scripted_model, tmp_path):
    replies = [reply(value) for value in (7, 3, 8, 2, 2, 6, 0, 3, 5, 1, 4, 9)]
    config = Config(evaluation_budget=12, initial_size=3, seed=28)
    uninterrupted = TraceAAD(task, scripted_model(replies), config=config)
    expected = uninterrupted.run()
    with TraceAAD(task, scripted_model(replies[:6]), config=config, run_dir=tmp_path) as first:
        for _ in range(6):
            first.step()
    resumed_model = scripted_model(replies[6:])
    resumed = TraceAAD(task, resumed_model, config=config, run_dir=tmp_path, resume=True)
    actual = resumed.run()
    assert actual.to_dict() == expected.to_dict()
    assert [p.to_dict() for p in resumed.archive.values()] == [p.to_dict() for p in uninterrupted.archive.values()]
    assert resumed_model.prompts == uninterrupted.model.prompts[6:]
    assert [(a.parent_id, a.operator, a.reference_id) for a in resumed.attempts] == [
        (a.parent_id, a.operator, a.reference_id) for a in uninterrupted.attempts]


@pytest.mark.parametrize("interrupted_stage", ["generate", "evaluate", "repair"])
def test_unfinished_attempt_is_dropped_on_resume(task, reply, scripted_model, tmp_path, interrupted_stage):
    responses = [KeyboardInterrupt()] if interrupted_stage == "generate" else [reply(body="return 1 / 0"), KeyboardInterrupt()]
    config = Config(evaluation_budget=2)
    first = TraceAAD(task, scripted_model(responses), config=config, run_dir=tmp_path)
    if interrupted_stage == "evaluate":
        def interrupt(code):
            raise KeyboardInterrupt()
        first.evaluator.evaluate = interrupt
    with pytest.raises(KeyboardInterrupt):
        first.step()
    resumed = TraceAAD(task, scripted_model([reply(2)] * 2), config=config, run_dir=tmp_path, resume=True)
    assert resumed.attempts == [] and resumed.archive == {}
    result = resumed.run()
    assert result.evaluations == 2 and result.best.evaluation.fitness == 2
