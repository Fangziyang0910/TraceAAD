"""V21.3: compact formation history and same-parent attempts, static fusion and free exploration."""
from dataclasses import replace
import json

import pytest

import traceaad_base
import traceaad_v21_3
from traceaad_base import Evaluation
from traceaad_base.models import Attempt
from traceaad_v21_2.prompts import PromptBuilder as PreviousPrompts
from traceaad_v21_3.context import REFINE, attempts, card, context, formation
from traceaad_v21_3.prompts import CANDIDATE_RESPONSE_FORMAT, PromptBuilder
from traceaad_v21_3.representation import failure
from traceaad_v21_3.search import PROTOCOL
from tests.v21_2.test_v21_2 import CONFIG, Reply, chain, code, program, reference_archive, task

REWRITE = "def solve(x):\n    s = 0\n    for i in range(3):\n        s += i * x\n    t = sorted([s, x])\n    return t[0]\n"
TRACEBACK = """Traceback (most recent call last):
  File "/evaluator/run.py", line 40, in evaluate
    value = fn(x)
  File "<candidate>", line 14, in get_dist
    return float(distance_matrix[tour[i]][tour[j]])
                                          ~~~~^^^
IndexError: list index out of range"""


def test_refine_selects_distinct_same_parent_attempts_across_operators():
    archive = chain(2)
    archive.update({i: program(i, score, 2) for i, score in ((3, 90.), (4, 99.), (5, 100.), (6, 101.))})
    archive[7] = program(7, 99., 2, source=archive[4].code + "\n# same implementation\n")
    archive[8] = program(8, 1., 2, operator="Crossover", reference=1)
    archive[9] = program(9, 1., 2, operator="Explore")
    archive[10] = program(10, 1., 1)
    archive[11] = program(11, None, 2, source=code(11, "x[0]"), failure=TRACEBACK)
    archive[12] = program(12, None, 2, source=archive[11].code, failure=TRACEBACK)
    archive[13] = program(13, None, 2, failure="ValueError: shape")
    archive[14] = program(14, None, 2, failure="TypeError: operand")
    valid, failed, total = attempts(archive, archive[2], CONFIG)
    assert [p.id for p in valid] == [7, 8, 9]
    assert [p.id for p in failed] == [13, 14]
    assert total == 11


def test_refine_without_improvement_keeps_recent_attempts_not_a_least_bad_exemplar():
    archive = chain(2)
    archive.update({i: program(i, 100. + i, 2) for i in range(3, 7)})
    valid, failed, _ = attempts(archive, archive[2], CONFIG)
    assert [p.id for p in valid] == [4, 5, 6] and failed == []


def test_refine_shows_formation_and_all_types_of_local_attempts_without_model_descriptions():
    archive = chain(12)
    archive[13] = program(13, 70., 12, analysis="Invented historical cause.")
    archive[14] = program(14, None, 12, source=code(14, "x[0]"), failure=TRACEBACK)
    archive[15] = program(15, 60., 12, operator="Explore")
    archive[16] = program(16, None, 12, operator="Crossover", reference=1, failure="ValueError: other")
    archive[17] = program(17, 1., 11)
    text, instruction, scores, view = context(archive, archive[12], "Refine", None, CONFIG)
    assert view["formation_path"] == list(range(1, 13))
    assert view["formation_steps"] == [2, 6, 7, 8, 9, 10, 11, 12]
    assert view["children"] == [13, 15] and view["failed"] == [14, 16]
    assert "Evaluated attempts: 4; showing 2 valid and 2 failed" in text
    assert "+    a = 13" in text and "+    return x[0] + a" in text and "Raised in get_dist:" in text
    assert "Attempt 1 (Refine): 88.0 -> 70.0; improved" in text
    assert "Attempt 3 (Explore): 88.0 -> 60.0; improved" in text
    assert "Attempt 4 (Crossover with reference Algorithm 1): Result: error" in text
    assert "Design:" not in text and "Invented" not in text and "Parent:" not in text and "Operator:" not in text
    assert text.count("```python") == 1
    assert text.index("[Evolution Path]") < text.index("[Attempts from Current Algorithm]") < text.index("[Current Algorithm]")
    assert instruction == REFINE and scores


def test_refine_without_local_attempts_keeps_formation_and_ignores_other_branches():
    archive = chain(12)
    archive[13] = program(13, 1., 11, operator="Explore")
    archive[14] = program(14, None, 11, operator="Crossover", failure=TRACEBACK, reference=1)
    assert context(archive, archive[12], "Refine", None, CONFIG) == \
        context(chain(12), archive[12], "Refine", None, CONFIG)
    text, _, _, view = context(archive, archive[12], "Refine", None, CONFIG)
    assert "[Evolution Path]" in text and "[Attempts from Current Algorithm]" not in text
    assert view["children"] == view["failed"] == []


def test_formation_selects_large_earlier_changes_and_the_last_three_including_current():
    archive = chain(12)
    archive[3].evaluation = Evaluation("success", 200.)
    selected, path = formation(archive, archive[12], 4)
    assert [p.id for p in path] == list(range(1, 13))
    assert [p.id for p in selected] == [4, 10, 11, 12]
    config = replace(CONFIG, ancestor_count=4)
    text, _, _, view = context(archive, archive[12], "Refine", None, config)
    assert "Initial: fitness = 100.0" in text
    assert "... 2 intermediate steps omitted." in text and "... 5 intermediate steps omitted." in text
    assert "Step 3: 200.0 -> 96.0; improved" in text
    assert "-    a = 3\n+    a = 4" in text
    assert "Step 11: 89.0 -> 88.0; improved" in text
    assert view["formation_steps"][-1] == archive[12].id


def test_formation_keeps_crossover_reference_without_repeated_parent_or_operator_fields():
    archive = chain(3)
    archive[2].operator, archive[2].reference_id = "Crossover", 99
    text, _, _, _ = context(archive, archive[3], "Refine", None, CONFIG)
    path = text.split("[Current Algorithm]")[0]
    assert "Step 1 (Crossover with reference Algorithm 99): 100.0 -> 98.0; improved" in path
    assert "Step 2: 98.0 -> 97.0; improved" in path
    assert "Parent:" not in path and "Operator:" not in path and "Design:" not in path
    assert "Algorithm 1\n" not in path and "Algorithm 2\n" not in path and "```python" not in path


def test_initial_program_has_no_fictitious_formation_change():
    archive = chain(1)
    text, _, _, view = context(archive, archive[1], "Refine", None, CONFIG)
    assert "[Evolution Path]" not in text
    assert view == {"formation_path": [1], "formation_steps": [], "children": [], "failed": []}


def test_formation_window_respects_zero_and_small_counts():
    archive = chain(6)
    for count, expected in ((0, []), (1, [6]), (2, [5, 6]), (3, [4, 5, 6])):
        selected, _ = formation(archive, archive[6], count)
        assert [p.id for p in selected] == expected
    text, _, _, _ = context(archive, archive[6], "Refine", None, replace(CONFIG, ancestor_count=0))
    assert "[Evolution Path]" not in text


def test_step_card_gives_result_before_complete_difference_without_design_or_analysis():
    parent = program(2, 98.)
    child = program(3, 90., 2)
    rows = card(parent, child, "Step 1").splitlines()
    assert rows[:2] == ["Step 1: 98.0 -> 90.0; improved", "Code change:"]
    assert "Design:" not in card(parent, child, "Step 1") and child.analysis not in card(parent, child, "Step 1")


def test_a_change_has_three_lines_of_context_and_a_rewrite_its_full_diff():
    before = "def solve(x):\n    a = 1\n    b = 2\n    c = 3\n    d = 4\n    e = 5\n    return a + b + c + d + e\n"
    parent = program(1, 10., operator="Init", source=before)
    child = program(2, 9., 1, source=before.replace("c = 3", "c = 30"))
    rows = card(parent, child, "Step 1").splitlines()
    start = rows.index("Code change:")
    assert rows[start + 1:start + 9] == [" def solve(x):", "     a = 1", "     b = 2", "-    c = 3", "+    c = 30",
                                         "     d = 4", "     e = 5", "     return a + b + c + d + e"]
    text = card(parent, program(3, 8., 1, source=REWRITE), "Step 1")
    assert "most of the code was replaced" not in text and "+        s += i * x" in text and "-    c = 3" in text


def test_identical_and_unparsable_code_are_stated():
    parent = program(2, 98.)
    child = program(3, 98., 2, source=parent.code + "\n# comment\n")
    assert "Code change: none; the normalized source is unchanged." in card(parent, child, "Step 1")
    child = program(4, None, 2, source="def solve(x:\n    return x\n", failure="SyntaxError: bad")
    text = card(parent, child, "Attempt 1")
    assert "the candidate code could not be parsed; the raw source texts are compared." in text
    assert "+def solve(x:" in text


def test_failure_gives_the_candidate_line_where_it_was_raised():
    assert failure(Evaluation("error", None, TRACEBACK)).splitlines() == [
        "Result: error; IndexError: list index out of range",
        "Raised in get_dist: return float(distance_matrix[tour[i]][tour[j]])"]
    assert failure(Evaluation("error", None, "ValueError: shape 3 mismatch")) == "Result: error; ValueError: shape N mismatch"
    assert failure(Evaluation("timeout", None, "Evaluation exceeded 10 seconds")) == \
        "Result: timeout; the evaluation did not finish within the allowed time."


@pytest.mark.parametrize("reference_code", [False, True])
def test_crossover_uses_only_current_program_and_reference_even_with_matching_history(reference_code):
    archive = reference_archive()
    archive[10] = program(10, 55., 9, operator="Crossover", reference=4)
    archive[11] = program(11, None, 2, operator="Crossover", reference=4, failure=TRACEBACK)
    archive[12] = program(12, 1., 9)
    actual = context(archive, archive[9], "Crossover", archive[4], CONFIG, reference_code=reference_code)
    assert actual == context({9: archive[9], 4: archive[4]}, archive[9], "Crossover", archive[4], CONFIG,
                             reference_code=reference_code)
    text, instruction, scores, view = actual
    reference, current = text.split("[Current Algorithm]")
    assert "Design: design_4" in reference and "Score: 30.0" in reference
    assert ("```python" in reference) == reference_code
    assert "return x + a + b - 1" in reference if reference_code else "return x + a + b - 1" not in reference
    assert "Design:" not in current and "Score: 60.0" in current and "```python" in current
    assert "Earlier" not in text and scores and view == {"reference_code": reference_code}
    assert "Otherwise, improve the current algorithm directly." in instruction


def test_crossover_without_reference_is_an_error():
    archive = chain(2)
    with pytest.raises(ValueError, match="requires a reference"):
        context(archive, archive[2], "Crossover", None, CONFIG)


def test_explore_uses_only_current_code_regardless_of_explorations_or_descendant_scores():
    archive = chain(12)
    archive[13] = program(13, 150., 12, operator="Explore")
    archive[14] = program(14, 1., 13)
    actual = context(archive, archive[12], "Explore", None, CONFIG)
    assert actual == context({12: archive[12]}, archive[12], "Explore", None, CONFIG)
    text, instruction, scores, view = actual
    assert "Design:" not in text and "Score" not in text and "Earlier" not in text
    assert "```python" in text and not scores and view == {}
    assert instruction == traceaad_base.prompts.EXPLORE


def test_initialization_and_all_repair_prompts_remain_identical():
    current, previous = PromptBuilder(task(), CONFIG), PreviousPrompts(task(), CONFIG)
    archive = chain(2)
    assert current.initialization(archive) == previous.initialization(archive)
    assert current.repair_format("{bad", "missing code") == previous.repair_format("{bad", "missing code")
    for evaluation in (Evaluation("error", None, TRACEBACK),
                       Evaluation("timeout", None, "Evaluation exceeded 10 seconds")):
        child = program(3, None, 2, failure=TRACEBACK)
        child.evaluation = evaluation
        assert current.repair_program(child) == previous.repair_program(child)
    assert list(CANDIDATE_RESPONSE_FORMAT["json_schema"]["schema"]["properties"]) == ["analysis", "code", "design"]


@pytest.mark.parametrize("old_protocol", ["refine_local_evidence_v1", "trajectory_design_reasoning_layered_v1"])
def test_search_persists_selected_records_and_resumes_only_the_same_protocol(tmp_path, old_protocol):
    config = traceaad_base.Config(evaluation_budget=20, initial_size=3, seed=5)
    run = traceaad_v21_3.TraceAAD(task(), Reply(), config=config, run_dir=tmp_path)
    assert run.run().evaluations == 20
    assert run.store.read("run_config.json")["generation_protocol"] == PROTOCOL
    for attempt in run.attempts:
        view = attempt.record.get("context")
        if attempt.operator == "Refine":
            assert all(run.archive[i].parent_id == attempt.parent_id
                       for i in view["children"] + view["failed"])
            path = view["formation_path"]
            assert path[-1] == attempt.parent_id
            assert all(run.archive[b].parent_id == a for a, b in zip(path, path[1:]))
        elif attempt.operator == "Explore":
            assert view == {}
        elif attempt.operator == "Crossover":
            assert view == {"reference_code": False}
    assert traceaad_v21_3.TraceAAD(task(), Reply(), config=config, run_dir=tmp_path, resume=True).evaluations_used == 20
    with pytest.raises(ValueError, match="generation_protocol"):
        traceaad_v21_3.TraceAAD(task(), Reply(), config=config, run_dir=tmp_path, resume=True, reference_code=True)
    saved = json.loads((tmp_path / "run_config.json").read_text())
    saved["generation_protocol"] = old_protocol
    (tmp_path / "run_config.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="generation_protocol"):
        traceaad_v21_3.TraceAAD(task(), Reply(), config=config, run_dir=tmp_path, resume=True)


def test_reference_source_control_is_persisted_and_reproduced_on_resume(tmp_path):
    run = traceaad_v21_3.TraceAAD(task(), Reply(), config=CONFIG, run_dir=tmp_path, reference_code=True)
    archive = reference_archive()
    run.archive = archive
    attempt = Attempt(10, 9, "Crossover", 4)
    prompt = run.evolution_prompt(attempt, archive[9], archive[4])
    assert attempt.record["context"] == {"reference_code": True}
    assert run.store.read("run_config.json")["generation_protocol"] == PROTOCOL + "+reference_code"
    resumed = traceaad_v21_3.TraceAAD(task(), Reply(), config=CONFIG, run_dir=tmp_path, resume=True, reference_code=True)
    resumed.archive = archive
    assert resumed.evolution_prompt(Attempt(10, 9, "Crossover", 4), archive[9], archive[4]) == prompt
