"""V10.19: programs and generation events, experience-weighted starting points, and Explore changes judged against the algorithm they were made to."""

import json
import random
import time

import pytest

from core import SecureEvaluator
from tests.support import TinyEvaluation, TokenLLM, response
from traceaad.v10_19 import Config, TraceAADV1019
from traceaad.common.delivery import parse_response
from traceaad.common.evaluation import SeededEvaluation
from traceaad.v10_19.selection import experience, sample_parent, temperature, weights
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


def method(tmp_path, *answers, budget=10, selection=None, **options):
    config = Config(budget=budget, **options)
    return TraceAADV1019(evaluation=TinyEvaluation(), selection_evaluation=selection,
                         llm=TokenLLM(*answers), run_dir=tmp_path, config=config)


def roots(m):
    for _ in range(9):
        m._roots()
    assert m.phase == "search" and len(m.archive) == 8


def test_evaluation_counts_outermost_calls_also_when_the_program_is_executed_again():
    seeded = SeededEvaluation(TinyEvaluation())
    evaluator = SecureEvaluator(seeded)
    recursive = "def score(x):\n    return x + 1 if x > 3 else score(x + 1)\n"
    seeded.reset()
    outcome = evaluator.evaluate_program_with_details(TinyEvaluation().template_program, source=recursive, seed=1)
    assert outcome.result == {"score": 5.0}
    assert seeded.measured()["calls"] == 1 and seeded.measured()["function_seconds"] >= 0
    again = SeededEvaluation(Reexecuting())
    again.reset()
    outcome = SecureEvaluator(again).evaluate_program_with_details(
        Reexecuting().template_program, source="def score(x):\n    return x\n", seed=1)
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
            '  File "/repo/traceaad/v10_19/probe.py", line 33, in call\n'
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
    assert set(diagnostics["actions"]) == {"Refine", "Explore", "Crossover", "Develop", "Repair"}
    records = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert any(r.get("program") and r.get("attempt") and r.get("progress") for r in records)
    resumed = method(tmp_path, *answers, budget=14, selection=SelectionEvaluation())
    assert resumed.phase == "finished" and resumed.programs.keys() == m.programs.keys()


# ---------- V10.19: an Explore change is judged against the algorithm it was made to ----------

def design(value):
    """A program scoring ``value`` whose code differs from the roots (which return integers)."""
    return f"Design: return {value}\n```python\ndef score(x):\n    return {value}\n```"


class Draws(random.Random):
    """Operator draws in a fixed order (the last one repeats)."""

    def __init__(self, *actions):
        super().__init__(0)
        self.actions = list(actions)

    def choices(self, population, weights=None, *, cum_weights=None, k=1):
        action = self.actions.pop(0) if len(self.actions) > 1 else self.actions[0]
        return [action] * k


def always(action="Explore"):
    return Draws(action)


def tags(m):
    return [(a["id"], a["parent_id"], a["action"], a["exploration"])
            for a in sorted(m.attempts_table.values(), key=lambda a: a["id"]) if a.get("exploration")]


def started(tmp_path, *answers, budget=40, **options):
    m = method(tmp_path, *(response(i) for i in range(1, 9)), *answers, budget=budget, **options)
    roots(m)
    m.action_rng = always()
    return m


def test_a_change_that_beats_its_algorithm_at_once_is_not_developed(tmp_path):
    m = started(tmp_path, design(9.5))
    m._search()
    attempts = m.attempts
    m._search()
    assert m.attempts == attempts
    record = m.facts.explorations[1]
    assert record["reason"] == "improved" and record["development_attempts"] == [] and record["improved_start"]
    assert record["start_id"] in m.archive and record["start_score"] == m.archive[record["start_id"]]["score"]


def test_a_change_behind_its_algorithm_is_developed_from_its_best_version_up_to_three_steps(tmp_path):
    m = started(tmp_path, design(0.5), design(0.7), design(0.6), design(0.65), design(0.9))
    for _ in range(6):  # proposal, three steps, the close, the next proposal
        m._search()
    assert [(aid, parent, action, e["step"]) for aid, parent, action, e in tags(m)] == [
        (9, tags(m)[0][1], "Explore", 0), (10, 9, "Develop", 1), (11, 10, "Develop", 2), (12, 10, "Develop", 3),
        (13, tags(m)[4][1], "Explore", 0)]
    record = m.facts.explorations[1]
    assert record["reason"] == "cap" and record["development_attempts"] == [10, 11, 12]
    assert (record["first_score"], record["best_score"], record["best_id"]) == (0.5, 0.7, 10)
    assert not record["improved_start"]


def test_development_ends_as_soon_as_a_version_beats_the_algorithm(tmp_path):
    m = started(tmp_path, design(0.5), design(0.6), design(9.5), design(0.8))
    for _ in range(5):  # proposal, two steps, the close, the next proposal
        m._search()
    record = m.facts.explorations[1]
    assert record["reason"] == "improved" and record["development_attempts"] == [10, 11]
    assert record["best_score"] == 9.5 and record["improved_start"]
    assert m.attempts_table[12]["action"] == "Explore" and m.attempts_table[12]["exploration"]["id"] == 2


def test_development_shares_the_explore_draws_and_blocks_new_proposals(tmp_path):
    m = method(tmp_path, *(response(i) for i in range(1, 9)), design(0.5), design(9.5), design(0.6), budget=40)
    roots(m)
    m.action_rng = Draws("Explore", "Refine", "Explore")
    m._search()  # proposal
    m._search()  # an ordinary Refine while the change is open
    m._search()  # the next Explore draw develops the change instead of proposing
    actions = [m.attempts_table[i]["action"] for i in (9, 10, 11)]
    assert actions == ["Explore", "Refine", "Develop"]
    assert m.attempts_table[10]["exploration"] is None and m.attempts_table[11]["exploration"] == {"id": 1, "step": 1}


def test_an_exploration_without_a_new_program_ends_at_its_proposal(tmp_path):
    m = started(tmp_path, response(8))
    m._search()
    assert m.attempts_table[9]["status"] == "duplicate"
    attempts = m.attempts
    m._search()
    assert m.attempts == attempts
    record = m.facts.explorations[1]
    assert record["proposed_id"] is None and record["reason"] == "no_program" and record["first_status"] == "duplicate"


def test_a_repaired_proposal_is_developed_from_its_repaired_version(tmp_path):
    m = started(tmp_path, BOOM, design(0.5), design(0.75))
    m._search()  # the proposal fails, its repair succeeds
    assert m.attempts_table[10]["repair_of"] == 9 and m.attempts_table[10]["status"] == "valid"
    request = m.prompts.develop(m.archive[10], m.programs[m.attempts_table[9]["parent_id"]], [])
    assert "is the first version of the change (its first version failed: runtime error" in request["prompt"]
    m._search()
    assert m.attempts_table[11]["parent_id"] == 10 and m.attempts_table[11]["exploration"] == {"id": 1, "step": 1}


def test_the_develop_context_shows_the_algorithm_the_change_was_made_to(tmp_path):
    m = started(tmp_path, design(0.5), design(0.7), design(0.6))
    for _ in range(3):
        m._search()
    opened = m._open_exploration()
    source = opened["source"]
    request = m.prompts.develop(opened["best"], source, opened["development"])
    prompt = request["prompt"]
    assert "[Current Algorithm]\nScore: 0.7" in prompt
    algorithm = prompt.split("[The Algorithm the Change Was Made To]")[1].split("[How the Change")[0]
    assert ("An Explore step changed this algorithm; the first version of the change scored 0.5 "
            "(worse than this algorithm). The best algorithm found so far in this search scores 8.") in algorithm
    assert f"Score: {source['score']:g}" in algorithm and f"return {source['score']:g}" in algorithm
    history = prompt.split("[How the Change Has Been Developed]")[1].split("[Other Attempts")[0]
    assert "Start · first version of the change · score 0.5" in history
    assert "Step 1 (latest: produced the current algorithm) · Develop · score 0.5 → score 0.7 (improved)" in history
    assert "1 development attempt did not produce a version on the path above. All are listed, oldest first" in prompt
    assert "Attempt 11 · Develop from the version scoring 0.7 · Design: return 0.6\n  → score 0.6 (worse)" in prompt
    assert ("Make the change work in the algorithm it was made to: keep the computation the change introduces "
            "and write a version that scores better than that algorithm. Every version and attempt shown above "
            "has already been evaluated.") in prompt
    assert "what the current algorithm computes differently from the algorithm the change was made to" in prompt
    assert request["attempt_ids"] == [11] and request["action"] == "Develop"


def test_a_change_is_one_attempt_of_its_algorithm_and_ends_in_its_best_version(tmp_path):
    m = started(tmp_path, design(0.5), design(0.6), design(9.5))
    m._search()
    source = m.programs[m.attempts_table[9]["parent_id"]]
    tried, improved = experience(m.attempts_table, m.programs, m.facts.explorations)
    assert (tried[source["id"]], improved[source["id"]]) == (1, 0)  # open: the first version is worse
    line = m.prompts.build("Refine", source)["prompt"].split("Attempt 9 · Explore")[1]
    assert "the change is being developed" in line.split("\n\n")[0]
    m._search()
    m._search()
    m._search()  # closes the change: its second development step beat the algorithm
    tried, improved = experience(m.attempts_table, m.programs, m.facts.explorations)
    assert (tried[source["id"]], improved[source["id"]]) == (1, 1)
    assert (tried[9], improved[9], tried[10], improved[10]) == (1, 1, 1, 1)  # development steps count where they start
    prompt = m.prompts.build("Refine", source)["prompt"]
    assert "1 attempt started from the current algorithm; 1 produced a new algorithm scoring better than it." in prompt
    assert ("→ score 0.5 (worse), evaluation time about 0.1 s; developed in 2 further steps: "
            "best version score 9.5 (improved)") in prompt


def test_a_resumed_run_continues_the_open_exploration(tmp_path):
    m = started(tmp_path, design(0.5), design(0.7))
    m._search()
    m._search()
    resumed = method(tmp_path, design(0.6), design(0.65), budget=40)
    resumed.action_rng = always()
    opened = resumed._open_exploration()
    assert opened["best"]["id"] == 10
    resumed._search()
    resumed._search()
    resumed._search()
    assert [(a["parent_id"], a["exploration"]["step"]) for a in resumed.attempts_table.values() if a["id"] > 10] == [
        (10, 2), (10, 3)]
    assert resumed.facts.explorations[1]["reason"] == "cap"


def test_the_end_of_the_budget_closes_an_open_exploration(tmp_path):
    m = started(tmp_path, design(0.5), design(0.7), budget=10)
    m._search()
    m._search()
    m._search()
    assert m.phase == "freeze"
    record = m.facts.explorations[1]
    assert record["reason"] == "budget" and record["development_attempts"] == [10]


def test_full_run_reports_exploration_diagnostics(tmp_path):
    answers = [response(i) for i in range(1, 9)] + [design(i / 100) for i in range(21, 60)]
    m = method(tmp_path, *answers, budget=40, selection=SelectionEvaluation())
    m.action_rng = always()
    summary = m.run()
    assert summary["status"] == "finished"
    from experiments.infra.diagnose_search import diagnose
    diagnose(tmp_path)
    explorations = json.loads((tmp_path / "diagnostics.json").read_text())["explorations"]
    # Each version scores a little more than the previous one, so a change made to a
    # root is developed for three steps and a change made to a developed version beats it.
    assert explorations["count"] >= 2 and explorations["developed"] >= 2
    assert explorations["development_generations"] + explorations["proposal_generations"] == 32
    assert set(explorations["reasons"]) <= {"cap", "improved", "budget"}
    assert explorations["development_steps_mean"] >= 1
    assert explorations["first_version_similarity_to_start_median"] is not None


def test_goals_aim_at_the_given_design_and_state_shown_versions_as_known(tmp_path):
    m = method(tmp_path, *(response(i) for i in range(1, 9)), budget=20)
    roots(m)
    refine = m.prompts.build("Refine", m.archive[3])["prompt"]
    assert ("Develop the current algorithm further: keep its core idea and write a version that scores better "
            "than it. Every version and attempt shown above has already been evaluated.") in refine
    explore = m.prompts.build("Explore", m.archive[3])["prompt"]
    assert ("Write an algorithm that scores better than the best found so far by changing how the current "
            "algorithm makes its decisions, not by tuning it. Keep unchanged the parts of the current algorithm "
            "that the change does not replace.") in explore
    crossover = m.prompts.build("Crossover", m.archive[3], reference=m.archive[8])["prompt"]
    assert "keep the current algorithm's core idea and write a version that scores better than it" in crossover
    with pytest.raises(ValueError):
        m.prompts.build("Develop", m.archive[3])
