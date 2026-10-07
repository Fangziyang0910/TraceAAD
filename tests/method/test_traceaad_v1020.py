"""V10.20: V10.17 with true facts about computation and a Deepen step that searches with the current rule."""

import json
import random

import numpy as np
import pytest

from benchmarks.tasks import HELDOUT_TIMEOUT, TASKS, training_task
from tests.support import TinyEvaluation, TokenLLM, response, small_task
from traceaad.common.selection import experience
from traceaad.v10_17 import Config as V1017Config, TraceAADV1017
from traceaad.v10_20 import Config, TraceAADV1020
from traceaad.v10_20.prompts import CALLED_ONCE, DEEPEN, SERVICE, PromptBuilder
from experiments.traceaad_v10_20.diagnose import diagnostics
from experiments.traceaad_v10_20.run import main as run_main

SELECTION_OFFSET = 100


class SelectionEvaluation(TinyEvaluation):
    def evaluate_program(self, program_str, callable_func, **kwargs):
        return callable_func(1) + SELECTION_OFFSET


class Always(random.Random):
    def __init__(self, action):
        super().__init__(0)
        self.action = action

    def choices(self, population, weights=None, *, cum_weights=None, k=1):
        return [self.action] * k


def method(tmp_path, *answers, budget=10, selection=None, cls=TraceAADV1020, config=Config):
    return cls(evaluation=TinyEvaluation(), selection_evaluation=selection,
               llm=TokenLLM(*answers), run_dir=tmp_path, config=config(budget=budget))


def roots(m):
    for _ in range(9):
        m._roots()
    assert m.phase == "search" and len(m.archive) == 8


def builder(task, evaluation=None):
    evaluation = evaluation or training_task(task, condition="traceaad")[0]
    return PromptBuilder(TokenLLM(), task, evaluation, {}, {}, Config())


# ---------- configuration ----------

def test_deepen_takes_its_share_from_the_three_v1017_steps():
    shares = Config().operators
    assert shares == {"Refine": 0.40, "Explore": 0.25, "Crossover": 0.20, "Deepen": 0.15}
    assert sum(shares.values()) == pytest.approx(1.0)
    v1017 = V1017Config()
    for name in ("Refine", "Explore", "Crossover"):
        assert v1017.operators[name] - shares[name] == pytest.approx(0.05)
    assert (Config().development_probability, Config().development_steps) == (0.125, 3)
    with pytest.raises(ValueError):
        Config(operators={"Refine": 0.5, "Develop": 0.5})


# ---------- Deepen ----------

def design(value):
    """A program scoring ``value`` whose code differs from the roots (which return integers)."""
    return f"Design: return {value}\n```python\ndef score(x):\n    return {value}\n```"


def test_deepen_decides_from_the_current_algorithm_and_earlier_deepen_attempts(tmp_path):
    m = method(tmp_path, *(response(i) for i in range(1, 9)), response(20), response(21), response(22), budget=14)
    roots(m)
    m._attempt(m.prompts.build("Refine", m.archive[8]), parent=m.archive[8])
    current = m.archive[9]
    m._attempt(m.prompts.build("Refine", current), parent=current)
    m._attempt(m.prompts.build("Deepen", current), parent=current)
    deepen = m.prompts.build("Deepen", current)
    prompt = deepen["prompt"]
    assert deepen["action"] == "Deepen" and DEEPEN in prompt and "[Your Task: Refine]" not in prompt
    assert ("Keep the current algorithm's decision rule and use it to guide a search that spends more of the time "
            "limit on finding better decisions, so that the program scores better than the current algorithm. "
            "Every version and attempt shown above has already been evaluated.") in prompt
    assert ("Analysis: <a few sentences: which decisions of the current algorithm a search guided by it could "
            "improve") in prompt
    # No formation path; of the attempts from the current algorithm only the Deepen one is shown.
    assert "[How the Current Algorithm Was Formed]" not in prompt and deepen["history_edge_ids"] == []
    assert "[Earlier Deepen Attempts From the Current Algorithm]" in prompt and deepen["attempt_ids"] == [11]
    assert ("1 earlier Deepen attempt started from the current algorithm; 1 produced a new algorithm scoring "
            "better than it. All are listed") in prompt
    assert "Attempt 11 · Deepen · Design: return 22" in prompt and "Attempt 10" not in prompt
    assert prompt.index("[Current Algorithm]") < prompt.index("[Earlier Deepen") < prompt.index("[Your Task: Deepen]")
    # Refine still sees every attempt, the Deepen one included.
    refine = m.prompts.build("Refine", current)
    assert refine["attempt_ids"] == [10, 11] and "[How the Current Algorithm Was Formed]" in refine["prompt"]
    first = m.prompts.build("Deepen", m.archive[3])
    assert "[Earlier Deepen Attempts" not in first["prompt"] and first["attempt_ids"] == []


def test_a_long_deepen_context_drops_the_oldest_deepen_attempts(tmp_path):
    m = method(tmp_path, *(response(i) for i in range(1, 9)), response(20), response(21), budget=14)
    roots(m)
    current = m.archive[8]
    m._attempt(m.prompts.build("Deepen", current), parent=current)
    m._attempt(m.prompts.build("Deepen", current), parent=current)
    full = m.prompts.build("Deepen", current)
    assert full["attempt_ids"] == [9, 10] and full["trims"] == []
    m.prompts.config.max_input_tokens = full["input_tokens"] - 1
    trimmed = m.prompts.build("Deepen", current)
    assert trimmed["attempt_ids"] == [10] and trimmed["trims"] == ["oldest_attempt"]
    assert "2 earlier Deepen attempts started from the current algorithm" in trimmed["prompt"]
    assert "The most recent 1 are listed" in trimmed["prompt"]


def test_earlier_versions_do_not_write_deepen(tmp_path):
    m = method(tmp_path, *(response(i) for i in range(1, 9)), cls=TraceAADV1017, config=V1017Config)
    roots(m)
    with pytest.raises(ValueError):
        m.prompts.build("Deepen", m.archive[8])


def test_a_deepen_proposal_counts_in_its_parent_experience_and_opens_an_exploration(tmp_path):
    m = method(tmp_path, *(response(i) for i in range(1, 9)), design(30.5), design(31.5), design(29.5),
               design(32.5), budget=20)
    m.config.development_probability = 1.0
    roots(m)
    m.action_rng = Always("Deepen")
    m._search()  # the proposal
    proposal = m.attempts_table[9]
    assert proposal["action"] == proposal["sampled_action"] == "Deepen"
    assert proposal["exploration"] == {"id": 1, "step": 0}
    for _ in range(4):  # three Refine steps from the best version reached, then the exploration closes
        m._search()
    steps = [(a["id"], a["action"], a["parent_id"], a["exploration"]) for a in m.attempts_table.values() if a["id"] > 9]
    assert steps == [(10, "Refine", 9, {"id": 1, "step": 1}), (11, "Refine", 10, {"id": 1, "step": 2}),
                     (12, "Refine", 10, {"id": 1, "step": 3})]
    record = m.facts.explorations[1]
    assert (record["first_score"], record["best_score"], record["development_attempts"]) == (30.5, 32.5, [10, 11, 12])
    tried, improved = experience(m.attempts_table, m.programs)
    assert tried[proposal["parent_id"]] == 1 and improved[proposal["parent_id"]] == 1
    prompt = m.prompts.build("Refine", m.programs[proposal["parent_id"]])["prompt"]
    assert "Attempt 9 · Deepen · Design: return 30.5" in prompt
    result = diagnostics(m.facts, 20, m.progress.init_attempts, 5, "v1020", 30)
    assert result["explorations"]["by_proposal_action"] == {
        "Deepen": {"count": 1, "developed": 1, "development_improved": 1}}


def test_full_run_reports_deepen_and_computation(tmp_path):
    answers = [response(i) for i in range(1, 9)] + [response(i) for i in range(20, 40)]
    m = method(tmp_path, *answers, budget=24, selection=SelectionEvaluation())
    summary = m.run()
    assert summary["status"] == "finished" and summary["method"] == "v1020"
    assert any(a["action"] == "Deepen" for a in m.attempts_table.values())
    result = diagnostics(m.facts, 24, summary["init_attempts"], 5, "v1020", 30)
    assert set(result["actions"]) == {"Refine", "Explore", "Crossover", "Deepen", "Repair"}
    computation = result["computation"]
    assert computation["time_limit"] == 30 and computation["clock_programs"] == 0
    assert 0 <= computation["best_time_share"] < 1
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert sum(e["kind"] == "candidate" for e in events) == 24


def test_the_run_entry_states_the_v1020_configuration(capsys):
    run_main(["--task", "op_aco", "--run-name", "dry", "--budget", "1000", "--dry-run"])
    shown = json.loads(capsys.readouterr().out)
    assert shown["method"] == "v1020" and shown["config"]["operators"]["Deepen"] == 0.15
    assert shown["search_timeout"] == 60


# ---------- facts about computation ----------

def test_a_task_without_benchmark_facts_keeps_the_v1017_text(tmp_path):
    new = method(tmp_path / "new", *(response(i) for i in range(1, 9)))
    old = method(tmp_path / "old", *(response(i) for i in range(1, 9)), cls=TraceAADV1017, config=V1017Config)
    assert new.prompts.common == old.prompts.common


@pytest.mark.parametrize("task", TASKS)
def test_each_task_states_its_training_and_test_conditions(task):
    prompts = builder(task)
    evaluation = prompts.common[1]
    train, _ = training_task(task, condition="traceaad")
    assert f"The whole evaluation must finish within {format(train.timeout_seconds, 'g')} seconds." in evaluation
    assert (f"each test set must finish within {format(HELDOUT_TIMEOUT[task], 'g')} seconds."
            in evaluation)
    expected = {
        "tsp_construct": ("16 training instances with 50 nodes", "16 instances each with 50, 100 and 200 nodes"),
        "cvrp_aco": ("10 training instances with 50 customers",
                     "64 instances each with 20, 50, 100 and 200 customers"),
        "op_aco": ("5 training instances with 50 nodes", "64 instances each with 50, 100 and 200 nodes"),
        "vrptw_construct": ("16 training instances with 50 customers",
                            "16 instances each with 50, 100 and 200 customers"),
        "online_bin_packing": ("4 training instances: 1,000 and 5,000 items, each with bin capacities 100 and 500",
                               "5 instances each of 1,000, 5,000 and 10,000 items with bin capacities 100 and 500"),
    }[task]
    assert f"Each program is evaluated on {expected[0]}." in evaluation
    assert f"After the search, the final program is tested on {expected[1]};" in evaluation
    program = {"score": 1.0, "eval_seconds": 2.04, "calls": 5, "function_seconds": 0.01}
    assert (f"Evaluation time: about 2.0 s of the {format(train.timeout_seconds, 'g')} s limit"
            in prompts.measured(program))


@pytest.mark.parametrize("task", ["cvrp_aco", "op_aco"])
def test_aco_text_says_the_function_is_called_once_per_instance(task):
    text = builder(task).common[0]
    assert "evaluated many" not in text and CALLED_ONCE in text
    evaluation = small_task(task)
    calls = []

    def heuristics(*args):
        calls.append(1)
        return np.ones_like(args[1] if task == "op_aco" else args[0])

    evaluation.evaluate_program("", heuristics)
    assert len(calls) == getattr(evaluation, "n_instance")


def test_vrptw_text_states_service_durations_as_the_evaluator_applies_them():
    assert builder("vrptw_construct").common[0].endswith(SERVICE)
    evaluation = small_task("vrptw_construct")
    seen = []

    def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, current_time,
                         demands, distance_matrix, time_windows):
        seen.append((current_node, current_time))
        return int(unvisited_nodes[0])

    evaluation.evaluate_program("", select_next_node)
    _, distance, _, _, service, windows = evaluation._datasets[0]
    # Along the first route, the time passed after serving a customer is its arrival,
    # raised to the window opening, plus its service duration.
    route = []
    for node, time in seen:
        if node == 0 and route:
            break
        route.append((node, time))
    for (previous, before), (node, after) in zip(route, route[1:]):
        arrival = before + distance[previous, node]
        assert after == pytest.approx(max(arrival, windows[node][0]) + service[node])
        assert service[node] > 0
