"""V10.16: programs and generation events, experience-weighted starting points, measured outcomes."""

import json
import random
import time

import pytest

from core import SecureEvaluator
from tests.support import TinyEvaluation, TokenLLM, response
from traceaad.v10_16 import Config, TraceAADV1016
from traceaad.common.delivery import parse_response
from traceaad.common.evaluation import SeededEvaluation
from traceaad.common.selection import experience, sample_parent, temperature, weights
from traceaad.common.evaluation import clean_traceback

BOOM = "Design: fails\n```python\ndef score(x):\n    raise ValueError('boom')\n```"


class SelectionEvaluation(TinyEvaluation):
    def __init__(self, offset=100, broken=()):
        super().__init__()
        self.offset, self.broken = offset, tuple(broken)

    def evaluate_program(self, program_str, callable_func, **kwargs):
        value = callable_func(1)
        if value in self.broken:
            raise ValueError("loops on the selection set")
        return value + self.offset


class Reexecuting(TinyEvaluation):
    """Executes the program text again for every instance, as OBP does."""

    def evaluate_program(self, program_str, callable_func, **kwargs):
        total = 0
        for value in range(3):
            namespace = {}
            exec(program_str, namespace)
            total += namespace["score"](value)
        return total


def method(tmp_path, *answers, budget=10, selection=None):
    return TraceAADV1016(evaluation=TinyEvaluation(), selection_evaluation=selection,
                         llm=TokenLLM(*answers), run_dir=tmp_path, config=Config(budget=budget))


def roots(m):
    for _ in range(9):
        m._roots()
    assert m.phase == "search" and len(m.archive) == 8


def test_evaluation_counts_outermost_calls_also_when_the_program_is_executed_again():
    seeded = SeededEvaluation(TinyEvaluation())
    evaluator = SecureEvaluator(seeded)
    recursive = "def score(x):\n    return x + 1 if x > 3 else score(x + 1)\n"
    seeded.reset()
    outcome = evaluator.evaluate_program_with_details(recursive, seed=1)
    assert outcome.result == {"score": 5.0}
    assert seeded.measured()["calls"] == 1 and seeded.measured()["function_seconds"] >= 0
    again = SeededEvaluation(Reexecuting())
    again.reset()
    outcome = SecureEvaluator(again).evaluate_program_with_details('def score(x):\n    return x\n', seed=1)
    assert outcome.result == {"score": 3.0} and again.measured()["calls"] == 3


def test_a_call_still_running_at_the_limit_is_measured_and_stated(tmp_path):
    seeded = SeededEvaluation(TinyEvaluation())
    seeded.reset()
    seeded.calls.value = 29
    seeded.call_started.value = time.monotonic() - 2.0
    measured = seeded.measured()
    assert measured["calls"] == 29 and measured["call_running"] and measured["function_seconds"] >= 2.0
    prompts = method(tmp_path, budget=1).prompts
    prompts.timeout = 30
    stuck = {"failure": {"kind": "timeout", "calls": 29, "function_seconds": 29.9, "call_running": True}}
    slow = {"failure": {"kind": "timeout", "calls": 5923, "function_seconds": 29.9, "call_running": False}}
    assert prompts.failure(stuck) == ("stopped at the 30 s time limit inside call 30 to the function, "
                                      "after 29 completed calls (about 29.9 s inside the function in total)")
    assert prompts.failure(slow) == ("stopped at the 30 s time limit after 5923 calls to the function, "
                                     "about 29.9 s inside it")


def test_tracebacks_drop_the_call_counter_frames():
    text = ('Traceback (most recent call last):\n'
            '  File "/repo/core/evaluate.py", line 320, in _evaluate_with_details\n'
            '    res = self._evaluator.evaluate_program(program_str, program_callable, **kwargs)\n'
            '  File "/repo/traceaad/v10_16/probe.py", line 33, in call\n'
            '    return function(*args, **kwargs)\n'
            '  File "<string>", line 2, in score\n'
            'ValueError: boom')
    cleaned = clean_traceback(text)
    assert "probe.py" not in cleaned and "return function(*args" not in cleaned
    assert cleaned.endswith('File "<string>", line 2, in score\nValueError: boom')


def test_delivery_uses_the_program_after_the_last_code_label():
    text = ("Analysis: the old version loops.\n```diff\n- a\n+ b\n```\nstray ``` fence\n"
            "Design: return two\nCode:\n```python\ndef score(x):\n    return 2\n```")
    code, idea, meta = parse_response(text, "stop", TinyEvaluation().template_program)
    assert "return 2" in code and idea == "return two"
    assert meta["strategy"].startswith("after_last_code_label:")


def test_temperature_stays_finite_when_many_programs_share_the_top_score():
    values = [2.0] * 10 + [1.0, 0.0] + [i / 10 for i in range(1, 10)]
    beta, ess = temperature(values)
    assert 0 < beta < float("inf") and ess == pytest.approx(8)
    assert temperature([1.0, 2.0, 3.0]) == (0.0, 3.0)
    nodes = [{"id": i, "fitness": v, "valid": True} for i, v in enumerate(values)]
    programs = {n["id"]: n for n in nodes}
    p, _ = weights(nodes, {}, programs)
    assert p[:10] == pytest.approx([p[0]] * 10)
    assert all(q > 0 for q in p[10:]) and p[10] < p[0]


def test_programs_tried_many_times_without_improving_lose_weight():
    programs = {1: {"id": 1, "fitness": 5.0, "valid": True}, 2: {"id": 2, "fitness": 5.0, "valid": True},
                3: {"id": 3, "fitness": 9.0, "valid": True}}
    attempts = {i: {"id": i, "action": "Refine", "parent_id": 1, "repair_of": None,
                    "status": "duplicate", "program_id": 2} for i in range(10, 20)}
    attempts[20] = {"id": 20, "action": "Explore", "parent_id": 2, "repair_of": None,
                    "status": "valid", "program_id": 3}
    tried, improved = experience(attempts, programs)
    assert (tried[1], improved[1], tried[2], improved[2]) == (10, 0, 1, 1)
    p, parts = weights([programs[1], programs[2]], attempts, programs)
    assert p[1] > 3 * p[0]
    assert parts["rate"] == pytest.approx(2 / 13)
    node, info = sample_parent([programs[1], programs[2]], attempts, programs, random.Random(0))
    assert info["tried"] in (10, 1) and info["eligible"] == 2


def test_a_repair_counts_for_the_attempt_it_repairs():
    programs = {1: {"id": 1, "fitness": 5.0, "valid": True}, 2: {"id": 2, "valid": False},
                3: {"id": 3, "fitness": 6.0, "valid": True}}
    attempts = {2: {"id": 2, "action": "Refine", "parent_id": 1, "repair_of": None,
                    "status": "timeout", "program_id": 2},
                3: {"id": 3, "action": "Repair", "parent_id": 2, "repair_of": 2,
                    "status": "valid", "program_id": 3}}
    tried, improved = experience(attempts, programs)
    assert tried == {1: 1} and improved == {1: 1}


def test_events_link_duplicates_failures_and_repairs(tmp_path):
    m = method(tmp_path, *(response(i) for i in range(1, 9)), response(3), BOOM, response(20), budget=12)
    roots(m)
    parent = m.archive[8]
    m._attempt(m.prompts.build("Refine", parent), parent=parent)
    duplicate = m.attempts_table[9]
    assert duplicate["status"] == "duplicate" and duplicate["program_id"] == 3 and not duplicate["program_id"]
    assert m.evaluation_calls == 8
    m._attempt(m.prompts.build("Refine", parent), parent=parent)
    failed, repair = m.attempts_table[10], m.attempts_table[11]
    assert failed["status"] == "runtime_error" and failed["program_id"] == 10 and m.programs[failed["program_id"]]["calls"] == 1
    assert not m.programs[10]["valid"] and "boom" in m.programs[10]["failure"]["error"]
    assert repair["repair_of"] == 10 and repair["parent_id"] == 10 and repair["action"] == "Repair"
    repaired = m.archive[11]
    assert repaired["parent_id"] == 10 and m.programs[repaired["program_id"]]["fitness"] == 20 and m.programs[repaired["program_id"]]["calls"] == 1
    # The formation path folds the failed first version into its repaired step:
    # the failure and its line are stated, the diff goes to the repaired version.
    request = m.prompts.build("Refine", repaired)
    prompt = request["prompt"]
    assert "Step 1 (latest: produced the current algorithm) · Refine, then Repair · score 8 → score 20 (improved)" in prompt
    assert "First version failed: runtime error: ValueError: boom at `raise ValueError('boom')`" in prompt
    assert "+    return 20" in prompt and "raise ValueError" not in prompt.split("Code diff")[1]
    assert request["history_edge_ids"] == [11]
    assert m.programs[10]["failure"]["line"] == "raise ValueError('boom')"
    # The next generation from root 8 sees what was already tried there.
    request = m.prompts.build("Refine", parent)
    assert request["attempt_ids"] == [9, 10]
    assert "2 attempts started from the current algorithm; 1 produced a new algorithm scoring better than it." in request["prompt"]
    assert "the same code as an algorithm already evaluated in this search (score 3)" in request["prompt"]
    assert ("failed: runtime error: ValueError: boom at `raise ValueError('boom')`; repaired: score 20 (improved)"
            in request["prompt"])
    assert "1 call to the function, under 0.1 s inside it" in m.prompts.measured(repaired)
    tried, improved = experience(m.attempts_table, m.programs)
    assert tried[8] == 2 and improved[8] == 1


def test_known_failure_links_to_the_failed_program_without_evaluation(tmp_path):
    m = method(tmp_path, *(response(i) for i in range(1, 9)), BOOM, BOOM, BOOM, budget=12)
    roots(m)
    parent = m.archive[1]
    m._attempt(m.prompts.build("Refine", parent), parent=parent)
    assert m.attempts_table[9]["status"] == "runtime_error"
    # The repair returns the same failing code: a known failure, not evaluated again.
    assert m.attempts_table[10]["status"] == "known_failure" and m.attempts_table[10]["program_id"] == 9
    calls = m.evaluation_calls
    m._attempt(m.prompts.build("Refine", parent), parent=parent)
    assert m.attempts_table[11]["status"] == "known_failure" and m.evaluation_calls == calls
    prompt = m.prompts.build("Refine", parent)["prompt"]
    assert "the repair returned the same failing program" in prompt
    assert ("the same code as a program that already failed (runtime error: ValueError: boom at "
            "`raise ValueError('boom')`)") in prompt


def test_progress_credits_a_repaired_program_to_the_generation_it_repaired(tmp_path):
    m = method(tmp_path, *(response(i) for i in range(1, 9)), BOOM, response(20), budget=12)
    roots(m)
    m._attempt(m.prompts.build("Refine", m.archive[8]), parent=m.archive[8])
    prompt = m.prompts.build("Explore", m.archive[2])["prompt"]
    assert "Attempt 9 · Refine, then Repair from an algorithm scoring 8 → 20 (previous best 8)" in prompt


def test_explore_sees_how_the_search_best_improved(tmp_path):
    m = method(tmp_path, *(response(i) for i in range(1, 9)), response(20), budget=10)
    roots(m)
    m._attempt(m.prompts.build("Refine", m.archive[8]), parent=m.archive[8])
    request = m.prompts.build("Explore", m.archive[2])
    prompt = request["prompt"]
    assert "[How the Best Score Improved in This Search]" in prompt
    assert "Attempt 9 · Refine from an algorithm scoring 8 → 20 (previous best 8)" in prompt
    assert "Attempt 8 · Init as an initial algorithm → 8 (previous best 7)" in prompt
    assert "Best score found so far in this search: 20." in prompt
    assert request["progress_ids"][-1] == 9 and len(request["progress_ids"]) == 8
    assert "[How the Current Algorithm Was Formed]" not in prompt


def test_a_finalist_failing_selection_is_replaced_by_the_next_program(tmp_path):
    m = method(tmp_path, *(response(i) for i in range(1, 9)), budget=8,
               selection=SelectionEvaluation(broken=(8,)))
    summary = m.run()
    assert summary["status"] == "finished"
    assert m.progress.finalists == [8, 7, 6, 5, 4, 3]
    selection = json.loads((tmp_path / "selection.json").read_text())
    assert selection["selected_node"] == 7 and selection["finalists"] == m.progress.finalists
    assert sum(r["fitness"] is None for r in selection["results"]) == 1


def test_full_run_reports_experience_diagnostics_and_resumes_identity(tmp_path):
    answers = [response(i) for i in range(1, 9)] + [response(i) for i in range(20, 26)]
    m = method(tmp_path, *answers, budget=14, selection=SelectionEvaluation())
    summary = m.run()
    assert summary["status"] == "finished" and summary["budget_used"] == 14
    from experiments.infra.diagnose_search import diagnose
    diagnose(tmp_path)
    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text())
    for name in ("attempts_on_tried_out_programs", "failure_rates_by_quarter", "explore_new_frontiers_after_300",
                 "failed_programs", "actions"):
        assert name in diagnostics
    assert set(diagnostics["actions"]) == {"Refine", "Explore", "Crossover", "Repair"}
    records = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert any(r.get("program") and r.get("attempt") and r.get("progress") for r in records)
    resumed = method(tmp_path, *answers, budget=14, selection=SelectionEvaluation())
    assert resumed.phase == "finished" and resumed.programs.keys() == m.programs.keys()
