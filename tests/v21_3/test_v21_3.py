"""V21.3: every evolution step shows its full actual code change; everything else is V21.2."""
import pytest

import traceaad_base
import traceaad_v21_3
from traceaad_v21_1.context import FAILURES
from traceaad_v21_2 import context as v21_2
from traceaad_v21_3.context import COMMON, CROSSOVER, EXPLORE, REFINE, card, context
from tests.v21_2.test_v21_2 import CONFIG, Reply, chain, code, program, reference_archive, task

REWRITE = "def solve(x):\n    s = 0\n    for i in range(3):\n        s += i * x\n    t = sorted([s, x])\n    return t[0]\n"


def test_instructions_and_selection_are_those_of_v21_2():
    assert (COMMON, REFINE, CROSSOVER, EXPLORE) == (v21_2.COMMON, v21_2.REFINE, v21_2.CROSSOVER, v21_2.EXPLORE)
    archive = chain(12)
    for i in range(13, 18):
        archive[i] = program(i, 90. - i, 12)
    archive[18] = program(18, None, 12, failure="ValueError: bad")
    for operator, reference in (("Refine", None), ("Crossover", archive[3]), ("Explore", None)):
        new = context(archive, archive[12], operator, reference, CONFIG)
        old = v21_2.context(archive, archive[12], operator, reference, CONFIG)
        assert new[1:] == old[1:]  # Instruction, score flag and the shown records are unchanged.


def test_small_change_has_three_lines_of_context():
    before = "def solve(x):\n    a = 1\n    b = 2\n    c = 3\n    d = 4\n    e = 5\n    return a + b + c + d + e\n"
    after = before.replace("c = 3", "c = 30")
    archive = {1: program(1, 10., operator="Init", source=before), 2: program(2, 9., 1, source=after)}
    rows = card(archive, archive[2]).splitlines()
    start = rows.index("Changed computation:")
    assert rows[start + 1:start + 9] == [" def solve(x):", "     a = 1", "     b = 2", "-    c = 3", "+    c = 30",
                                         "     d = 4", "     e = 5", "     return a + b + c + d + e"]


def test_a_rewrite_shows_its_full_diff_not_a_summary():
    archive = {1: program(1, 50., operator="Init", source=code(1)), 2: program(2, 20., 1, source=REWRITE,
                                                                                 operator="Explore")}
    text = card(archive, archive[2])
    assert "most of the code was replaced" not in text
    assert "+        s += i * x" in text and "-    a = 1" in text
    assert text.splitlines()[-1] == "Result relative to Algorithm 1: 50.0 -> 20.0; improved"


def test_failed_offspring_show_their_change_and_error():
    archive = chain(2)
    archive[3] = program(3, None, 2, source=code(3, "x[0]"), failure="IndexError: invalid index")
    text, instruction, _, view = context(archive, archive[2], "Refine", None, CONFIG)
    failed = text[text.index("[Failed Offspring]"):text.index("[Current Algorithm]")]
    assert "+    return x[0] + a" in failed and "Result: error; IndexError: invalid index" in failed
    assert view["failed"] == [3] and instruction.endswith(FAILURES)


def test_unparsable_code_is_compared_as_raw_text_and_says_so():
    archive = chain(2)
    archive[3] = program(3, None, 2, source="def solve(x:\n    return x\n", failure="SyntaxError: bad")
    text = card(archive, archive[3])
    assert "the code of Algorithm 3 could not be parsed; the raw source texts are compared." in text
    assert "+def solve(x:" in text


def test_identical_code_says_none():
    archive = chain(2)
    archive[3] = program(3, 98., 2, source=archive[2].code + "\n# comment\n")
    assert "Changed computation: none; the code is identical to Algorithm 2." in card(archive, archive[3])


def test_reference_formation_uses_the_same_step_cards():
    archive = {1: program(1, 50., operator="Init", source=code(1)), 2: program(2, 20., 1, source=REWRITE,
                                                                                 operator="Explore")}
    archive[3] = program(3, 70., operator="Init", source=code(3))
    text, _, _, view = context(archive, archive[3], "Crossover", archive[2], CONFIG)
    part = text[text.index("[Reference Algorithm]"):text.index("[Current Algorithm]")]
    assert view["reference_steps"] == [2] and "```python" not in part
    assert "Added lines that remain" not in part and "most of the code was replaced" not in part
    assert card(archive, archive[2]) in part
    archive = reference_archive()
    text, _, _, _ = context(archive, archive[9], "Crossover", archive[4], CONFIG, reference_code=True)
    part = text[text.index("[Reference Algorithm]"):text.index("[Current Algorithm]")]
    assert "```python" in part and "+    b = x * 3" in part


def test_ancestors_show_changes_and_the_initial_algorithm_its_design():
    archive = chain(4)
    text, _, _, _ = context(archive, archive[4], "Refine", None, CONFIG)
    history = text[text.index("[Ancestor History]"):text.index("[Current Algorithm]")]
    assert "design_1" in history and history.count("Changed computation:") == 2
    assert "Decision 1." not in history


def test_search_records_its_protocol_and_rejects_other_resumes(tmp_path):
    config = traceaad_base.Config(evaluation_budget=20, initial_size=3, seed=5)
    model = Reply()
    run = traceaad_v21_3.TraceAAD(task(), model, config=config, run_dir=tmp_path / "a")
    assert run.run().evaluations == 20
    assert run.store.read("run_config.json")["generation_protocol"] == "trajectory_design_reasoning_full_diff_v1"
    assert all("most of the code was replaced" not in p for p in model.prompts)
    assert traceaad_v21_3.TraceAAD(task(), Reply(), config=config, run_dir=tmp_path / "a",
                                   resume=True).evaluations_used == 20
    with pytest.raises(ValueError, match="generation_protocol"):
        traceaad_v21_3.TraceAAD(task(), Reply(), config=config, run_dir=tmp_path / "a", resume=True,
                                reference_code=True)
    control = traceaad_v21_3.TraceAAD(task(), Reply(), config=config, run_dir=tmp_path / "b", reference_code=True)
    assert control.store.read("run_config.json")["generation_protocol"] == \
        "trajectory_design_reasoning_full_diff_v1+reference_code"
