"""V21.3: layered evidence (design, actual change, full code) chosen by each material's role."""
import pytest

import traceaad_base
import traceaad_v21_3
from traceaad_base import Evaluation
from traceaad_v21_1.context import FAILURES
from traceaad_v21_2 import context as v21_2
from traceaad_v21_3.context import COMMON, CROSSOVER, EXPLORE, REFINE, card, context, integrations
from traceaad_v21_3.representation import failure
from tests.v21_2.test_v21_2 import CONFIG, Reply, chain, code, program, reference_archive, task

REWRITE = "def solve(x):\n    s = 0\n    for i in range(3):\n        s += i * x\n    t = sorted([s, x])\n    return t[0]\n"
TRACEBACK = """Traceback (most recent call last):
  File "/evaluator/run.py", line 40, in evaluate
    value = fn(x)
  File "<candidate>", line 14, in get_dist
    return float(distance_matrix[tour[i]][tour[j]])
                                          ~~~~^^^
IndexError: list index out of range"""


def test_instructions_differ_from_v21_2_in_the_material_note_and_crossover_history():
    assert (REFINE, EXPLORE) == (v21_2.REFINE, v21_2.EXPLORE)
    assert COMMON == ("Code differences and evaluation results are observed facts. Designs describe implementations "
                      "but may be inaccurate. A score change alone does not establish why the result changed.")
    assert "formation records" not in CROSSOVER and "earlier attempts from the current algorithm" not in CROSSOVER
    assert "earlier Crossover attempts with the reference algorithm" in CROSSOVER
    assert "what problems those integrations exposed" in CROSSOVER
    assert "end with one sentence that states the change you make" in CROSSOVER


def test_no_card_shows_text_derived_from_analysis():
    archive = chain(6)
    for i in range(7, 10):
        archive[i] = program(i, 90. - i, 6, analysis=f"Long reasoning {i}. Final sentence {i}.")
    archive[10] = program(10, None, 6, failure="ValueError: bad", analysis="Plan ten. Final ten.")
    for operator, reference in (("Refine", None), ("Crossover", archive[3])):
        text, _, _, _ = context(archive, archive[6], operator, reference, CONFIG)
        assert "Decision" not in text and "Final" not in text and "Reasoning" not in text and "reasoning" not in text


def test_step_card_gives_the_result_before_design_and_change():
    archive = chain(2)
    archive[3] = program(3, 90., 2)
    rows = card(archive, archive[3]).splitlines()
    assert rows[:5] == ["Algorithm 3", "Parent: Algorithm 2", "Operator: Refine",
                        "Result relative to Algorithm 2: 98.0 -> 90.0; improved", "Design: design_3"]
    assert rows.index("Code change:") == 5
    failed = program(4, None, 2, source=code(4, "x[0]"), failure=TRACEBACK)
    rows = card({**archive, 4: failed}, failed).splitlines()
    assert rows[3:6] == ["Result: error; IndexError: list index out of range",
                         "Raised in get_dist: return float(distance_matrix[tour[i]][tour[j]])", "Design: design_4"]


def test_a_change_has_three_lines_of_context_and_a_rewrite_its_full_diff():
    before = "def solve(x):\n    a = 1\n    b = 2\n    c = 3\n    d = 4\n    e = 5\n    return a + b + c + d + e\n"
    archive = {1: program(1, 10., operator="Init", source=before),
               2: program(2, 9., 1, source=before.replace("c = 3", "c = 30")),
               3: program(3, 8., 1, source=REWRITE, operator="Explore")}
    rows = card(archive, archive[2]).splitlines()
    start = rows.index("Code change:")
    assert rows[start + 1:start + 9] == [" def solve(x):", "     a = 1", "     b = 2", "-    c = 3", "+    c = 30",
                                         "     d = 4", "     e = 5", "     return a + b + c + d + e"]
    text = card(archive, archive[3])
    assert "most of the code was replaced" not in text and "+        s += i * x" in text and "-    c = 3" in text


def test_identical_and_unparsable_code_are_stated():
    archive = chain(2)
    archive[3] = program(3, 98., 2, source=archive[2].code + "\n# comment\n")
    assert "Code change: none; the code is identical to Algorithm 2." in card(archive, archive[3])
    archive[4] = program(4, None, 2, source="def solve(x:\n    return x\n", failure="SyntaxError: bad")
    text = card(archive, archive[4])
    assert "the code of Algorithm 4 could not be parsed; the raw source texts are compared." in text
    assert "+def solve(x:" in text


def test_failure_gives_the_candidate_line_where_it_was_raised():
    assert failure(Evaluation("error", None, TRACEBACK)).splitlines() == [
        "Result: error; IndexError: list index out of range",
        "Raised in get_dist: return float(distance_matrix[tour[i]][tour[j]])"]
    assert failure(Evaluation("error", None, "ValueError: shape 3 mismatch")) == "Result: error; ValueError: shape N mismatch"
    assert failure(Evaluation("timeout", None, "Evaluation exceeded 10 seconds")) == \
        "Result: timeout; the evaluation did not finish within the allowed time."


def test_failed_offspring_show_design_change_and_where_they_failed():
    archive = chain(2)
    archive[3] = program(3, None, 2, source=code(3, "x[0]"), failure=TRACEBACK)
    text, instruction, _, view = context(archive, archive[2], "Refine", None, CONFIG)
    failed = text[text.index("[Failed Offspring]"):text.index("[Current Algorithm]")]
    assert "Design: design_3" in failed and "+    return x[0] + a" in failed
    assert "Raised in get_dist:" in failed and view["failed"] == [3] and instruction.endswith(FAILURES)


def test_nearest_three_ancestors_show_changes_and_farther_ones_only_designs():
    archive = chain(12)
    text, _, _, view = context(archive, archive[12], "Refine", None, CONFIG)
    history = text[text.index("[Ancestor History]"):text.index("[Current Algorithm]")]
    cards = history.split("\n\nAlgorithm ")[1:]
    assert view["ancestor_changes"] == [9, 10, 11]
    for row in cards:
        number = int(row.split("\n")[0])
        assert ("Code change:" in row) == (number in (9, 10, 11))
        assert f"Design: design_{number}" in row
    assert "Decision" not in history


def test_current_and_reference_algorithms_show_design_and_full_code():
    archive = reference_archive()
    text, instruction, _, view = context(archive, archive[9], "Crossover", archive[4], CONFIG)
    reference = text[text.index("[Reference Algorithm]"):text.index("[Current Algorithm]")]
    assert reference.splitlines()[1:4] == ["Algorithm 4", "Score: 30.0", "Design: design_4"]
    assert "```python" in reference and "return x + a + b - 1" in reference
    assert "Formation" not in reference and "Parent:" not in reference
    current = text[text.index("[Current Algorithm]"):]
    assert "Design: design_9" in current and "Score: 60.0" in current and "```python" in current
    assert instruction == "\n".join([COMMON, CROSSOVER]) and "reference_steps" not in view
    assert "[Earlier Crossovers" not in text and "[Evaluated Offspring]" not in text and view["integrations"] == []


def crossover_archive():
    archive = reference_archive()
    archive[10] = program(10, 55., 9, operator="Crossover", reference=4)            # current with this reference
    archive[11] = program(11, 58., 9, operator="Crossover", reference=2)            # current with another reference
    archive[12] = program(12, 50., 9, operator="Refine")                            # current, not a Crossover
    archive[13] = program(13, 35., 3, operator="Crossover", reference=4)            # another receiver, this reference
    archive[14] = program(14, None, 2, operator="Crossover", reference=4, failure=TRACEBACK)
    archive[15] = program(15, 42., 2, operator="Crossover", reference=4)
    archive[16] = program(16, 41., 1, operator="Crossover", reference=4)
    return archive


def test_crossover_shows_only_earlier_crossovers_with_the_reference_current_algorithm_first():
    archive = crossover_archive()
    shown, total = integrations(archive, archive[9], archive[4], CONFIG)
    assert total == 5 and [p.id for p in shown] == [10, 14, 15, 16]  # 3 valid (10 first, then latest), 1 failed
    text, instruction, _, view = context(archive, archive[9], "Crossover", archive[4], CONFIG)
    block = text[text.index("[Earlier Crossovers with the Reference]"):text.index("[Reference Algorithm]")]
    assert "used Algorithm 4 as reference: 5; showing 4, those from Algorithm 9 first" in block
    assert "Algorithm 11\n" not in block and "Algorithm 12\n" not in block and "Algorithm 13\n" not in block
    assert "Parent: Algorithm 2\nOperator: Crossover\nReference: Algorithm 4\nResult: error" in block
    assert "Raised in get_dist:" in block and "Code change:" in block
    assert "[Evaluated Offspring]" not in text and "[Failed Offspring]" not in text
    assert view["integrations"] == [10, 14, 15, 16] and view["children"] == [] and view["failed"] == []
    assert instruction == "\n".join([COMMON, CROSSOVER, FAILURES])


def test_explore_current_algorithm_has_design_and_code_but_no_score():
    archive = chain(3)
    archive[4] = program(4, 120., 2, operator="Explore")
    text, instruction, scores, view = context(archive, archive[4], "Explore", None, CONFIG)
    current = text[text.index("[Current Algorithm]"):]
    assert "Design: design_4" in current and "Score" not in current and instruction == EXPLORE
    assert view["explorations"] == [1]


def test_search_records_its_protocol_and_rejects_reference_code(tmp_path):
    config = traceaad_base.Config(evaluation_budget=20, initial_size=3, seed=5)
    model = Reply()
    run = traceaad_v21_3.TraceAAD(task(), model, config=config, run_dir=tmp_path / "a")
    assert run.run().evaluations == 20
    assert run.store.read("run_config.json")["generation_protocol"] == "trajectory_design_reasoning_layered_v1"
    assert all("Decision:" not in p and "most of the code was replaced" not in p for p in model.prompts)
    assert traceaad_v21_3.TraceAAD(task(), Reply(), config=config, run_dir=tmp_path / "a",
                                   resume=True).evaluations_used == 20
    with pytest.raises(TypeError):
        traceaad_v21_3.TraceAAD(task(), Reply(), config=config, run_dir=tmp_path / "b", reference_code=True)
