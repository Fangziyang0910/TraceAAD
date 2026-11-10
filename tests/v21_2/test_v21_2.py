"""V21.2: trajectory-guided design reasoning in analysis, operator material for that reasoning."""
import json

import pytest

from traceaad_base import Evaluation, Program, Task
import traceaad_base
import traceaad_v21_2
from traceaad_v21_1.context import FAILURES
from traceaad_v21_2.context import (COMMON, CROSSOVER, EXPLORE, REFINE, card, context, decision, explorations,
                                    reference_steps)
from traceaad_v21_2.prompts import CANDIDATE_RESPONSE_FORMAT, COMMON_TEMPLATE

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


def test_analysis_task_is_set_by_the_instruction():
    assert list(CANDIDATE_RESPONSE_FORMAT["json_schema"]["schema"]["properties"]) == ["analysis", "code", "design"]
    line = next(l for l in COMMON_TEMPLATE.splitlines() if l.startswith('- "analysis"'))
    assert "according to [Instruction]" in line and "2–4 sentences" in line and "120 words" in line
    assert "approach you will implement" not in COMMON_TEMPLATE
    for text in (REFINE, CROSSOVER, EXPLORE):
        assert "end with one sentence that states the change you make" in text
        for word in ("TSP", "route", "tour", "2-opt", "schedule", "colour", "pheromone", "time limit", "efficient"):
            assert word not in text


def test_decision_is_the_last_sentence_before_any_repair():
    p = program(2, 1., analysis="Opportunity here. History says this. I replace the score with a rollout.")
    assert decision(p) == "I replace the score with a rollout."
    p.analysis = "First idea. Final change!\nRepair: fixed an index error."
    assert decision(p) == "Final change!"
    p.analysis = "Use a cheaper estimate, e.g. a partial rollout. Replace the scan with i.e. a heap."
    assert decision(p) == "Replace the scan with i.e. a heap."
    p.analysis = None
    assert decision(p) == ""


def test_offspring_cards_show_decision_change_result_and_no_reasoning_or_design():
    archive = chain(2)
    archive[3] = program(3, 90., 2, analysis="Long reasoning about history. Use a lookahead.")
    text = card(archive, archive[3], change=True)
    rows = text.splitlines()
    assert "Decision: Use a lookahead." in text and "Long reasoning" not in text and "design_3" not in text
    assert rows.index("Decision: Use a lookahead.") < rows.index("Changed computation:") < len(rows) - 1
    assert rows[-1] == "Result relative to Algorithm 2: 98.0 -> 90.0; improved"


def test_refine_without_offspring_reads_formation_and_code_only():
    archive = chain(4)
    text, instruction, scores, view = context(archive, archive[4], "Refine", None, CONFIG)
    assert "[Evaluated Offspring]" not in text and "[Ancestor History]" in text
    assert "design_2" not in text and "Decision 2." in text
    assert "design_1" in text and "Decision 1." not in text  # The initial algorithm shows its design, not a decision.
    assert instruction == "\n".join([COMMON, REFINE]) and scores
    assert "otherwise, reason from the current computation and the task" in REFINE
    archive[5] = program(5, 90., 4)
    archive[6] = program(6, None, 4, failure="ValueError: bad")
    text, instruction, _, view = context(archive, archive[4], "Refine", None, CONFIG)
    assert view["children"] == [5] and view["failed"] == [6]
    assert instruction == "\n".join([COMMON, REFINE, FAILURES])


def reference_archive():
    archive = {1: program(1, 50., operator="Init", source="def solve(x):\n    a = 1\n    return x + a\n")}
    archive[2] = program(2, 40., 1, source="def solve(x):\n    a = 1\n    b = x * 3\n    return x + a + b\n")
    archive[3] = program(3, 45., 2, source="def solve(x):\n    a = 1\n    b = x * 3\n    c = 0\n    return x + a + b\n")
    archive[4] = program(4, 30., 3, source="def solve(x):\n    a = 1\n    b = x * 3\n    return x + a + b - 1\n")
    archive[9] = program(9, 60., operator="Init", source="def solve(x):\n    return 2 * x\n")
    return archive


def test_reference_formation_keeps_steps_whose_lines_survive():
    archive = reference_archive()
    steps, surviving = reference_steps(archive, archive[4])
    assert [p.id for p in steps] == [2, 4]  # Step 3 added `c = 0`, which the reference no longer has.
    assert surviving(archive[2]) == {"b = x * 3"} and surviving(archive[3]) == set()
    assert surviving(archive[4]) == {"return x + a + b - 1"}


def test_crossover_shows_reference_design_and_formation_without_full_code_by_default():
    archive = reference_archive()
    text, instruction, _, view = context(archive, archive[9], "Crossover", archive[4], CONFIG)
    part = text[text.index("[Reference Algorithm]"):text.index("[Current Algorithm]")]
    assert "design_4" in part and "```python" not in part
    assert "+    b = x * 3" in part and "Decision 2." in part and "c = 0" not in part.split("Formation")[0]
    assert view["reference_steps"] == [2, 4]
    assert instruction == "\n".join([COMMON, CROSSOVER])
    assert "Align input representations and numerical scales" in CROSSOVER
    text, _, _, _ = context(archive, archive[9], "Crossover", archive[4], CONFIG, reference_code=True)
    part = text[text.index("[Reference Algorithm]"):text.index("[Current Algorithm]")]
    assert "```python" in part and "return x + a + b - 1" in part


def test_a_rewrite_step_lists_the_introduced_lines_the_reference_keeps():
    archive = {1: program(1, 50., operator="Init", source="def solve(x):\n    a = 1\n    return x + a\n")}
    rewrite = "def solve(x):\n    s = 0\n    for i in range(3):\n        s += i * x\n    t = sorted([s, x])\n    return t[0]\n"
    archive[2] = program(2, 20., 1, source=rewrite, operator="Explore")
    text, _, _, view = context({**archive, 3: program(3, 70., operator="Init", source=code(3))},
                               program(3, 70., operator="Init", source=code(3)), "Crossover", archive[2], CONFIG)
    assert view["reference_steps"] == [2]
    assert "most of the code was replaced" in text and "Introduced lines that Algorithm 2 still executes" in text
    assert "s += i * x" in text


def test_crossover_offspring_prefer_earlier_integrations():
    archive = chain(2)
    archive[3] = program(3, 90., 2, operator="Crossover", reference=1)
    for i in range(4, 8):
        archive[i] = program(i, 90. - i, 2)
    _, _, _, view = context(archive, archive[2], "Crossover", archive[1], CONFIG)
    assert view["children"] == [3, 6, 7]


def test_explore_reads_earlier_explorations_as_facts():
    archive = chain(3)
    archive[4] = program(4, 120., 2, operator="Explore")
    archive[5] = program(5, 80., 4)
    archive[6] = program(6, 70., 5)
    archive[7] = program(7, 110., 3, operator="Explore", source=archive[4].code)  # Same source: shown once.
    archive[8] = program(8, 130., 3, operator="Explore")
    rows, total, shown = explorations(archive, archive[8])
    assert total == 2 and shown == [1, 4]
    four = rows[1]
    assert "Score at creation: 120.0 (best in the search then: 97.0)" in four
    assert "Development: 2 later Refine or Crossover attempts from it and its descendants; best score among them: 70.0" in four
    assert "design_4" in four
    text, instruction, scores, view = context(archive, archive[8], "Explore", None, CONFIG)
    assert text.index("[Earlier Explorations]") < text.index("[Current Algorithm]") and scores
    assert instruction == EXPLORE and view["explorations"] == [1, 4]
    current = text[text.index("[Current Algorithm]"):]
    assert "Score" not in current and "design_8" not in current
    for verdict in ("untested", "insufficient", "under-explored"):
        assert verdict not in text


def test_explore_without_history_shows_only_the_code():
    archive = {1: program(1, 5., operator="Init")}
    text, _, scores, view = context(archive, archive[1], "Explore", None, CONFIG)
    assert text.startswith("[Current Algorithm]") and not scores and view["explorations"] == []


def test_search_runs_resumes_and_records_the_reference_condition(tmp_path):
    config = traceaad_base.Config(evaluation_budget=24, initial_size=3, seed=3)
    model = Reply()
    run = traceaad_v21_2.TraceAAD(task(), model, config=config, run_dir=tmp_path / "a")
    assert run.run().evaluations == 24
    assert run.store.read("run_config.json")["generation_protocol"] == "trajectory_design_reasoning_v1"
    assert any("Decision: Change" in p for p in model.prompts)
    assert all("Why " not in p for p in model.prompts if "[Evaluated Offspring]" in p)
    explore = [p for p in model.prompts if EXPLORE in p]
    assert explore and any("[Earlier Explorations]" in p for p in explore)
    resumed = traceaad_v21_2.TraceAAD(task(), Reply(), config=config, run_dir=tmp_path / "a", resume=True)
    assert resumed.evaluations_used == 24
    with pytest.raises(ValueError, match="generation_protocol"):
        traceaad_v21_2.TraceAAD(task(), Reply(), config=config, run_dir=tmp_path / "a", resume=True,
                                reference_code=True)
    control = traceaad_v21_2.TraceAAD(task(), Reply(), config=config, run_dir=tmp_path / "b", reference_code=True)
    assert control.store.read("run_config.json")["generation_protocol"].endswith("+reference_code")


def test_refine_singles_out_the_best_attempt_only_when_it_improved():
    archive = chain(2)
    archive[3] = program(3, 99., 2)  # Least-bad regression from 98.
    for i in range(4, 7):
        archive[i] = program(i, 120. + i, 2)
    _, _, _, view = context(archive, archive[2], "Refine", None, CONFIG)
    assert view["children"] == [4, 5, 6]
    archive[3].evaluation = Evaluation("success", 90.)
    text, _, _, view = context(archive, archive[2], "Refine", None, CONFIG)
    assert view["children"] == [3, 5, 6] and "the best improvement, if any, and the most recent" in text
