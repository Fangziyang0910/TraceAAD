"""V21.3: compact modification, integration and exploration histories for the three operators."""
from dataclasses import replace
import json

import pytest

import traceaad_base
import traceaad_v21_3
from traceaad_base import Evaluation, Program, Task
from traceaad_base.models import Attempt
from traceaad_v21_3.context import EXPLORATIONS, EXPLORE, REFINE, attempts, card, context, explorations, formation, integrations
from traceaad_v21_3.prompts import CANDIDATE_RESPONSE_FORMAT, PromptBuilder
from traceaad_v21_3.representation import failure
from traceaad_v21_3.search import PROTOCOL

CONFIG = traceaad_base.Config()


def code(k, body="x"):
    return f"def solve(x):\n    a = {k}\n    return {body} + a\n"


def program(i, score, parent=0, *, source=None, operator="Refine", failure=None, analysis=None, reference=None):
    evaluation = Evaluation("error", None, failure) if failure else Evaluation("success", score)
    return Program(i, source or code(i), f"design_{i}", operator, parent, reference, evaluation,
                   analysis=analysis or f"Reasoning {i}. Decision {i}.")


def chain(n):
    archive = {1: program(1, 100., operator="Init")}
    for i in range(2, n + 1):
        archive[i] = program(i, 100. - i, i - 1)
    return archive


def task():
    return Task("Return a finite score; lower is better.", "def solve(x):\n    return 1\n",
                lambda _, fn: float(fn(0)), score_definition="lower is better",
                entrypoint="solve", time_limit=5., evaluation_seed=1)


class Reply:
    def __init__(self):
        self.prompts = []

    def complete(self, prompt, **_):
        self.prompts.append(prompt)
        k = len(self.prompts)
        lines = "\n    ".join(f"v{i} = x + {i}" for i in range(k % 5 + 1))
        return json.dumps({"analysis": f"Why {k}. Change {k}.",
                           "code": f"def solve(x):\n    {lines}\n    return {k % 7} + x\n", "design": f"d{k}"})


def reference_archive():
    archive = {1: program(1, 50., operator="Init", source="def solve(x):\n    a = 1\n    return x + a\n")}
    archive[2] = program(2, 40., 1, source="def solve(x):\n    a = 1\n    b = x * 3\n    return x + a + b\n")
    archive[3] = program(3, 45., 2, source="def solve(x):\n    a = 1\n    b = x * 3\n    c = 0\n    return x + a + b\n")
    archive[4] = program(4, 30., 3, source="def solve(x):\n    a = 1\n    b = x * 3\n    return x + a + b - 1\n")
    archive[9] = program(9, 60., operator="Init", source="def solve(x):\n    return 2 * x\n")
    return archive


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


def test_old_improvement_does_not_override_the_recent_attempt_window():
    archive = chain(2)
    archive[3] = program(3, 1., 2)
    archive.update({i: program(i, 100. + i, 2) for i in range(4, 7)})
    valid, _, _ = attempts(archive, archive[2], CONFIG)
    assert [p.id for p in valid] == [4, 5, 6]


def test_refine_shows_formation_and_all_types_of_local_attempts_without_model_descriptions():
    archive = chain(12)
    archive[13] = program(13, 70., 12, analysis="Invented historical cause.")
    archive[14] = program(14, None, 12, source=code(14, "x[0]"), failure=TRACEBACK)
    archive[15] = program(15, 60., 12, operator="Explore")
    archive[16] = program(16, None, 12, operator="Crossover", reference=1, failure="ValueError: other")
    archive[17] = program(17, 1., 11)
    text, instruction, scores, view = context(archive, archive[12], "Refine", None, CONFIG)
    assert view["formation_path"] == list(range(1, 13))
    assert view["formation_steps"] == [5, 6, 7, 8, 9, 10, 11, 12]
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


def test_formation_keeps_recent_changes_instead_of_selecting_by_score_movement():
    archive = chain(12)
    archive[3].evaluation = Evaluation("success", 200.)
    selected, path = formation(archive, archive[12], 4)
    assert [p.id for p in path] == list(range(1, 13))
    assert [p.id for p in selected] == [9, 10, 11, 12]
    config = replace(CONFIG, ancestor_count=4)
    text, _, _, view = context(archive, archive[12], "Refine", None, config)
    assert "Initial: fitness = 100.0" in text
    assert "... 7 intermediate steps omitted." in text
    assert "Step 8: 92.0 -> 91.0; improved" in text
    assert "-    a = 8\n+    a = 9" in text
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


def integration_archive():
    return {1: program(1, 100., operator="Init"), 2: program(2, 90., operator="Init"),
            3: program(3, 80., 2), 4: program(4, 95., 1),
            5: program(5, 94., 4, operator="Crossover", reference=1),
            6: program(6, 85., 2, operator="Crossover", reference=3),
            7: program(7, 99., 1, operator="Crossover", reference=2),
            8: program(8, 70., 4),
            9: program(9, None, 4, operator="Crossover", reference=1, failure=TRACEBACK)}


@pytest.mark.parametrize("reference_code", [False, True])
def test_crossover_shows_formation_and_receiver_and_reference_integrations(reference_code):
    archive = integration_archive()
    text, instruction, scores, view = context(archive, archive[4], "Crossover", archive[3], CONFIG,
                                             reference_code=reference_code)
    assert view == {"reference_path": [2, 3], "reference_steps": [3], "integrations": [5, 6, 9],
                    "reference_code": reference_code}
    assert "[Reference Evolution Path]" in text and "Step 1: 90.0 -> 80.0; improved" in text
    assert "Integration 1 from Algorithm 4 (Crossover with reference Algorithm 1): 95.0 -> 94.0; improved" in text
    assert "Integration 2 from Algorithm 2 (Crossover with reference Algorithm 3): 90.0 -> 85.0; improved" in text
    assert "Reference design: design_1" in text and "Receiver design: design_2" in text
    assert "Raised in get_dist:" in text
    reference = text.split("[Reference Algorithm]")[1].split("[Current Algorithm]")[0]
    assert "Design: design_3" in reference and ("```python" in reference) == reference_code
    assert "Program" not in text and "Parent:" not in text and scores
    assert "otherwise improve the current algorithm directly" in instruction


def test_crossover_preserves_equal_outcomes_from_different_receivers():
    archive = integration_archive()
    archive[10] = program(10, 85., 1, source=archive[6].code, operator="Crossover", reference=3)
    valid, failed, total = integrations(archive, archive[4], archive[3], CONFIG)
    assert [p.id for p in valid] == [5, 6, 10] and [p.id for p in failed] == [9] and total == 4


def test_crossover_without_reference_is_an_error():
    archive = chain(2)
    with pytest.raises(ValueError, match="requires a reference"):
        context(archive, archive[2], "Crossover", None, CONFIG)


def test_explore_shows_directions_best_implementations_and_paths_without_crediting_later_explore():
    archive = {1: program(1, 20., operator="Init"), 2: program(2, 25., operator="Init"),
               3: program(3, 18., 1), 4: program(4, 12., 3, operator="Crossover", reference=2),
               5: program(5, 30., 4, operator="Explore"), 6: program(6, 10., 5),
               7: program(7, None, 1, failure=TRACEBACK), 8: program(8, 27., 2, operator="Explore"),
               9: program(9, 25., source=code(2), operator="Init")}
    archive[4].analysis = "Invented historical cause."
    rows, total, shown = explorations(archive)
    assert total == 4 and [r["origin"] for r in shown] == [1, 5, 8, 9]
    assert shown[0] == {"origin": 1, "development_count": 3, "best": 4, "best_path": [1, 3, 4]}
    assert shown[1] == {"origin": 5, "development_count": 1, "best": 6, "best_path": [5, 6]}
    assert "Best later design: design_4" in rows[0]
    assert "Refine -> Crossover with reference Algorithm 2" in rows[0]
    assert "including the origin = 12.0" in rows[0] and "including the origin = 10.0" not in rows[0]
    assert "Best later design:" not in rows[-1]
    text, instruction, scores, view = context(archive, archive[8], "Explore", None, CONFIG)
    assert view == {"explorations": shown} and scores and instruction == EXPLORE
    assert text.count("```python") == 1 and "Invented historical cause" not in text
    current = text.split("[Current Algorithm]")[1]
    assert "Score:" not in current and "Design:" not in current


def test_exploration_window_follows_recent_development_as_well_as_recent_birth():
    archive = {i: program(i, 100. + i, operator="Init") for i in range(1, EXPLORATIONS + 3)}
    child = EXPLORATIONS + 3
    archive[child] = program(child, 200., 1)
    rows, total, shown = explorations(archive)
    assert total == EXPLORATIONS + 2 and len(rows) == EXPLORATIONS
    assert [r["origin"] for r in shown] == [1, 4, 5, 6, 7, 8]
    assert shown[0]["best"] == 1 and shown[0]["best_path"] == [1]


def test_equal_origins_keep_the_branch_with_recent_development():
    archive = {1: program(1, 20., operator="Init"), 2: program(2, 20., source=code(1), operator="Init"),
               3: program(3, 10., 1)}
    _, total, shown = explorations(archive)
    assert total == 1
    assert shown == [{"origin": 1, "development_count": 1, "best": 3, "best_path": [1, 3]}]


def test_explore_without_prior_directions_has_only_current_code():
    parent = program(1, 10.)
    text, instruction, scores, view = context({1: parent}, parent, "Explore", None, CONFIG)
    assert "[Earlier Explorations]" not in text and not scores
    assert view == {"explorations": []} and instruction == EXPLORE


def test_initialization_and_repair_keep_the_common_template():
    prompts = PromptBuilder(task(), CONFIG)
    archive = chain(2)
    child = program(3, None, 2, failure=TRACEBACK)
    for text in (prompts.initialization(archive), prompts.repair_format("{bad", "missing code"),
                 prompts.repair_program(child)):
        assert "explain the approach you will implement" in text
        assert "Available evolution records" not in text
    assert "Available evolution records" in prompts.evolution_render("ctx", "do", has_scores=False)
    assert list(CANDIDATE_RESPONSE_FORMAT["json_schema"]["schema"]["properties"]) == ["analysis", "code", "design"]


@pytest.mark.parametrize("old_protocol", ["refine_evolution_history_v1", "refine_local_evidence_v1",
                                         "trajectory_design_reasoning_layered_v1"])
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
            for direction in view["explorations"]:
                path = direction["best_path"]
                assert path[0] == direction["origin"] and path[-1] == direction["best"]
                assert all(run.archive[b].parent_id == a for a, b in zip(path, path[1:]))
                assert all(run.archive[i].operator != "Explore" for i in path[1:])
        elif attempt.operator == "Crossover":
            assert view["reference_code"] is False and view["reference_path"][-1] == attempt.reference_id
            assert all(run.archive[i].operator == "Crossover" and
                       (run.archive[i].parent_id == attempt.parent_id or run.archive[i].reference_id == attempt.reference_id)
                       for i in view["integrations"])
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
    assert attempt.record["context"]["reference_code"] is True
    assert attempt.record["context"]["reference_steps"] == [2, 3, 4]
    assert run.store.read("run_config.json")["generation_protocol"] == PROTOCOL + "+reference_code"
    resumed = traceaad_v21_3.TraceAAD(task(), Reply(), config=CONFIG, run_dir=tmp_path, resume=True, reference_code=True)
    resumed.archive = archive
    assert resumed.evolution_prompt(Attempt(10, 9, "Crossover", 4), archive[9], archive[4]) == prompt
