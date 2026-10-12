"""V10.17 feedback and prompt roles, explicit repair history, and isolated recovery."""

import json

import pytest

from benchmarks.tasks import CO_TASKS, training_task
from tests.support import TinyEvaluation, TokenLLM, response
from traceaad.common.prompts import ContextTooLong
from traceaad.v10_17.prompts import PromptBuilder as V1017Prompts
from traceaad.v10_17.traceaad import TraceAADV1017
from traceaad.v10_21.prompts import PromptBuilder as ProtocolPrompts
from traceaad.v10_25 import Config, TraceAADV1025
from traceaad.v10_25.prompts import PromptBuilder


def method(tmp_path, *answers, **options):
    return TraceAADV1025(evaluation=TinyEvaluation(), llm=TokenLLM(*answers),
                        run_dir=tmp_path, config=Config(**options))


def start(search):
    search._roots()
    search._roots()
    return search.archive[1]


def propose(search, action, parent):
    return search._attempt(search.prompts.build(action, parent), parent=parent)


def test_v1017_feedback_reduces_weight_after_unproductive_attempts(tmp_path):
    search = method(tmp_path, response(1), response(1), response(1), response(1),
                    budget=5, roots=1)
    parent = start(search)
    # Compare the same source before and after three duplicate attempts.
    _, before = search._choose_parent([parent])
    for _ in range(3):
        propose(search, "Refine", parent)
    _, after = search._choose_parent([parent])
    assert before["tried"] == 0 and after["tried"] == 3
    assert after["improved"] == 0 and after["prospect"] < before["prospect"]
    assert "beta" in after and "rank" not in after
    assert search.attempts == 4 and len(search.archive) == 1


def test_prompt_changes_preserve_materials_and_untouched_operators(tmp_path):
    search = method(tmp_path, response(1), response(2), response(3), budget=5, roots=1)
    parent = start(search)
    child = propose(search, "Refine", parent)
    propose(search, "Explore", parent)
    baseline = ProtocolPrompts(search.llm, search.task, search.training.evaluation,
                               search.programs, search.attempts_table, search.config)
    assert search.config.attempts_shown == search.config.progress_shown == 8
    assert search.prompts.REFINE == V1017Prompts.REFINE
    assert search.prompts.EXPLORE == V1017Prompts.EXPLORE
    assert search.prompts.initial([]) == baseline.initial([])
    assert search.prompts.initial([parent] * 4) == baseline.initial([parent] * 4)
    for action in ("Refine", "Explore", "Crossover"):
        kwargs = {"reference": search.archive[3]} if action == "Crossover" else {}
        actual = search.prompts.build(action, child, **kwargs)
        expected = baseline.build(action, child, **kwargs)
        assert {k: v for k, v in actual.items() if k not in {"prompt", "input_tokens"}} == {
            k: v for k, v in expected.items() if k not in {"prompt", "input_tokens"}}
        revised = expected["prompt"].replace(baseline.ANALYSIS[action], search.prompts.ANALYSIS[action])
        for subject in ("current", "reference"):
            revised = revised.replace(
                baseline.FORMATION_INTRO.replace("current algorithm", f"{subject} algorithm"),
                search.prompts.FORMATION_INTRO.replace("current algorithm", f"{subject} algorithm"))
        revised = revised.replace(baseline.CROSSOVER, search.prompts.CROSSOVER)
        assert actual["prompt"] == revised
        assert actual["input_tokens"] == search.llm.count_prompt_tokens(revised)
        if action == "Explore":
            assert actual == expected
    explore = search.prompts.build("Explore", parent)
    assert not explore["history_edge_ids"] and explore["progress_ids"] == [2, 3]
    assert explore["attempt_ids"] == [2, 3]
    assert "Best score found so far" in explore["prompt"]
    assert "scores better than the best found so far" in explore["prompt"]
    assert "SEARCH" not in explore["prompt"]


def test_root_and_weaker_reference_have_no_assumed_advantage(tmp_path):
    search = method(tmp_path, response(10), response(1), budget=4, roots=1)
    parent = start(search)
    weaker = propose(search, "Explore", parent)
    assert weaker["fitness"] > parent["fitness"]
    refine = search.prompts.build("Refine", parent)["prompt"]
    assert "no changes have been recorded yet" in refine
    assert "any available code changes and evaluated outcomes" in refine
    cross = search.prompts.build("Crossover", parent, reference=weaker)["prompt"]
    assert "what the reference algorithm does well" not in cross
    assert "Adapt a computation from the reference algorithm that could improve" in cross
    assert "how to adapt it to work with the current computation" in cross
    assert "Keep the current algorithm's core idea" in cross


def test_failed_proposal_and_repair_are_distinct_diffs_in_one_step(tmp_path):
    failed = "Design: try missing computation\nCode:\n```python\ndef score(x):\n    return missing(x)\n```"
    search = method(tmp_path, response(1), failed, response(2), budget=5, roots=1, history_depth=1)
    parent = start(search)
    repaired = propose(search, "Explore", parent)
    prompt = search.prompts.build("Refine", repaired)
    assert prompt["history_edge_ids"] == [3]
    assert "program 1 → failed program 2" in prompt["prompt"]
    assert "failed program 2 → program 3" in prompt["prompt"]
    assert "+    return missing(x)" in prompt["prompt"]
    assert "-    return missing(x)" in prompt["prompt"]
    assert "Failed proposal's Design: try missing computation" in prompt["prompt"]
    assert "scores report the results of whole programs" in prompt["prompt"]
    repair_prompt = search.prompts.repair(search.programs[2])["prompt"]
    assert "retaining the intended decision computation" in repair_prompt
    assert "return missing(x)" in repair_prompt and "[Error]" in repair_prompt
    assert "Code diff (previous → current)" not in prompt["prompt"]
    cross = search.prompts.build("Crossover", parent, reference=repaired)
    assert cross["reference_history_edge_ids"] == [3]
    assert "program 1 → failed program 2" in cross["prompt"]
    assert "failed program 2 → program 3" in cross["prompt"]
    # Preserve the base mechanism's explicit fallback instead of silently losing half a repair.
    search.config.max_input_tokens = 1
    with pytest.raises(ContextTooLong):
        search.prompts.build("Refine", repaired)


def test_full_code_only_and_no_free_repair_at_budget_end(tmp_path):
    edits = "Edits:\n<<<<<<< SEARCH\nreturn 1\n=======\nreturn 2\n>>>>>>> REPLACE"
    failed = "Code:\n```python\ndef score(x):\n    return missing(x)\n```"
    search = method(tmp_path, response(1), edits, failed, budget=3, roots=1)
    parent = start(search)
    propose(search, "Refine", parent)
    propose(search, "Explore", parent)
    assert [a["status"] for a in search.attempts_table.values()] == [
        "valid", "delivery_failed", "runtime_error"]
    assert search.attempts == 3 and search.progress.repair_id is None
    assert len(search.llm.calls) == 3


def test_trimming_keeps_both_diffs_of_latest_repair(tmp_path):
    failed = "Design: try a computation\n```python\ndef score(x):\n    return missing(x)\n```"
    search = method(tmp_path, response(1), response(2), failed, response(3), budget=6, roots=1)
    root = start(search)
    child = propose(search, "Refine", root)
    repaired = propose(search, "Explore", child)
    full = search.prompts.build("Refine", repaired)
    assert full["history_edge_ids"] == [2, 4]
    search.config.max_input_tokens = full["input_tokens"] - 1
    trimmed = search.prompts.build("Refine", repaired)
    assert trimmed["trims"] == ["oldest_history"]
    assert trimmed["history_edge_ids"] == [4]
    assert "program 2 → failed program 3" in trimmed["prompt"]
    assert "failed program 3 → program 4" in trimmed["prompt"]
    assert repaired["code"].rstrip() in trimmed["prompt"]
    assert trimmed["input_tokens"] <= search.config.max_input_tokens


def test_weak_exploration_continues_from_its_best_and_resumes(tmp_path):
    config = dict(budget=5, roots=1, operators={"Explore": 1.}, development_probability=1.)
    search = method(tmp_path, response(10), response(1), **config)
    start(search)
    search._search()
    assert search.archive[2]["fitness"] > search.archive[1]["fitness"]
    assert all(search._developed(i) == TraceAADV1017._developed(search, i) for i in range(30))
    resumed = method(tmp_path, response(2), response(0), response(3), **config)
    assert resumed.run()["status"] == "finished"
    assert [resumed.attempts_table[i]["parent_id"] for i in (3, 4, 5)] == [2, 3, 3]
    assert resumed.facts.explorations[1]["development_attempts"] == [3, 4, 5]
    selected = json.loads((tmp_path / "selection.json").read_text())
    assert selected["selected_key"] == resumed.archive[1]["key"]
    with pytest.raises(ValueError, match="same configuration"):
        method(tmp_path, **{**config, "seed": 1})


@pytest.mark.parametrize("old_mechanism", ["v1025-trajectories-1", "v1025-trajectories-2",
                                         "v1025-v1017-base-1", None])
def test_abandoned_checkpoints_cannot_resume_or_be_modified(tmp_path, old_mechanism):
    search = method(tmp_path, response(1), budget=3, roots=1)
    search._roots()
    checkpoint = tmp_path / "resume.json"
    saved = json.loads(checkpoint.read_text())
    assert saved["state"]["identity"]["mechanism"] == "v1025-v1017-base-2"
    if old_mechanism is None:
        del saved["state"]["identity"]["mechanism"]
    else:
        saved["state"]["identity"]["mechanism"] = old_mechanism
    checkpoint.write_text(json.dumps(saved))
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="mechanism"):
        method(tmp_path, budget=3, roots=1)
    assert before == {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("task", CO_TASKS)
def test_six_task_protocol_and_unambiguous_score_direction(task):
    evaluation = training_task(task, condition="traceaad")[0]
    prompts = PromptBuilder(TokenLLM(), task, evaluation, {}, {}, Config())
    initial = prompts.initial([])
    assert initial["input_tokens"] <= Config().max_input_tokens
    assert "Lower is better" in initial["prompt"]
    assert "SEARCH" not in initial["prompt"]
    assert "time limit" not in initial["prompt"]
    assert Config().eval_timeout_seconds == 120


def test_launcher_uses_a_new_batch_and_keeps_eighteen_runs():
    from experiments.traceaad_v10_25.launch_host import plan_for
    from experiments.traceaad_v10_25.run import build_parser
    plan = plan_for("v1025_v1017base_check")
    assert len(plan) == 18 and {row["task"] for row in plan} == set(CO_TASKS)
    assert all(row["seed"] == row["repeat"] - 1 for row in plan)
    args = build_parser().parse_args(["--task", "tsp_construct", "--dry-run"])
    assert args.experiment == "traceaad_v10_25"
