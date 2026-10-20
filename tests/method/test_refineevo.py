"""Offline checks for the RefineEvo loop: planner, operator refinement, experience pool, budget and recovery."""

import json
import re

import numpy as np
import pytest

from baselines.refineevo import RefineEvo
from baselines.refineevo.planner import Planner
from baselines.refineevo.refineevo import OPERATOR_DESCRIPTIONS
from baselines.shared.profiler import ProfilerBase
from core import Evaluation
from core.code import parse_program
from experiments.infra.resume_baseline import recover
from tests.support import TokenLLM
from traceaad.storage import read_json, rows, write_json


TEMPLATE = 'def solve(x: float) -> float:\n    """Return a value."""\n    return 9.0\n'
GENERATOR = "You are a world-class expert in optimization algorithms"
PLANNER = "You are an expert meta-optimization strategist"
CHECKER = "You are an expert algorithm evolution analyst"
IMPROVER = "You are an expert in evolutionary algorithm design and meta-optimization"
REFLECTOR = "You are an expert algorithmic researcher"


class ToyTask(Evaluation):
    def __init__(self):
        super().__init__(TEMPLATE, "Minimize the returned value.", safe_evaluate=False)
        self.evaluated = []

    def evaluate_program(self, code, function, **kwargs):
        self.evaluated.append(code)
        value = float(function(0))
        if value < 0:
            raise RuntimeError("PRIVATE_FAILURE_MESSAGE")
        return value


def program(value, name="solve"):
    return f"{{Return {value}.}}\n```python\ndef {name}(x: float) -> float:\n    return {value}\n```"


def plan(*operators, strategy="exploitation"):
    return "```json\n" + json.dumps([{"strategy": strategy, "reasoning": "r", "selected_operators": list(operators),
                                      "confidence": 0.9}]) + "\n```"


class Scripted(TokenLLM):
    """Answer each RefineEvo prompt by its role; the nth generated program returns values(n)."""

    def __init__(self, values=None, *, generator=None, planner=None, check=None, improve=None,
                 stop_after=None, fail_first=False):
        super().__init__()
        self.values = values or (lambda n: 5.0 - 0.01 * n)
        self.generator = generator or (lambda n: program(self.values(n)))
        self.planner = planner or (lambda n: plan("op3"))
        self.check = check or (lambda n: '```json\n{"needs_improvement": false, "reasoning": "fine", '
                                         '"confidence": 0.5, "improvement_urgency": "low"}\n```')
        self.improve = improve or (lambda n: '```json\n{"improvement_type": "radical", "reasoning": "stuck", '
                                             f'"new_description": "IMPROVED{n}", "confidence": 0.8}}\n```')
        self.stop_after, self.fail_first = stop_after, fail_first
        self.counts = {}
        self.stages = []

    def _next(self, stage):
        self.stages.append(stage)
        self.counts[stage] = self.counts.get(stage, 0) + 1
        return self.counts[stage]

    def draw_sample_with_details(self, prompt, **kwargs):
        if self.stop_after is not None and len(self.calls) >= self.stop_after:
            raise KeyboardInterrupt
        self.calls.append(prompt)
        if self.fail_first and len(self.calls) == 1:
            raise ValueError("Offline transport failure")
        system = prompt[0]["content"]
        if system.startswith(GENERATOR):
            content = self.generator(self._next("generate"))
        elif system.startswith(PLANNER):
            content = self.planner(self._next("planner"))
        elif system.startswith(CHECKER):
            content = self.check(self._next("check"))
        elif system.startswith(IMPROVER):
            content = self.improve(self._next("improve"))
        else:
            assert system.startswith(REFLECTOR)
            n = self._next("reflect")
            content = ("```json\n" + json.dumps([{"summary": f"LESSON{n}", "recommendations": ["r"],
                                                  "applicable_when": "w"}]) + "\n```")
        return {"content": content, "model": "offline",
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}


class Embedding:
    """Deterministic query vectors; `fail_at` raises on that call."""

    def __init__(self, fail_at=None):
        self.queries, self.fail_at = [], fail_at
        self.last_usage = {"total_tokens": 1}

    def __call__(self, query):
        self.queries.append(query)
        if self.fail_at is not None and len(self.queries) >= self.fail_at:
            raise RuntimeError("Offline embedding failure")
        return [1.0, float(len(query) % 7), float(query.count("objective"))]

    def close(self):
        pass


def method(directory, llm, *, budget=40, embedding=None, **options):
    task = ToyTask()
    write_json(directory / "run_config.json", {"method": "refineevo", "task": "toy", "budget": budget,
                                                "method_params": {}})
    np.random.seed(3)
    import random
    random.seed(3)
    options = {"pop_size": 2, "init_pop_size": 4, **options}
    return RefineEvo(llm, task, embedding or Embedding(), ProfilerBase(directory), max_sample_nums=budget,
                     **options), task


def candidates(directory):
    return [r for r in rows(directory / "events.jsonl") if r["kind"] == "candidate"]


def events(directory, name=None):
    found = [r["data"] for r in rows(directory / "events.jsonl") if r["kind"] == "method"]
    return [e for e in found if name is None or e["event"] == name]


def users(llm, stage):
    return [c[1]["content"] for c, s in zip(llm.calls, llm.stages) if s == stage]


def test_prompts_follow_upstream_roles_and_order(tmp_path):
    llm = Scripted()
    # 4 initial, then per generation one planner call and 2 op3 offspring; the last batch ends the budget.
    instance, task = method(tmp_path, llm, budget=8)
    instance.run()
    assert llm.stages == (["generate"] * 4 + ["planner"] + ["generate"] * 2 + ["reflect"] * 2
                          + ["planner"] + ["generate"] * 2)
    system, init = llm.calls[0][0]["content"], llm.calls[0][1]["content"]
    assert "**Function Name:** solve" in system and "Minimize the returned value." in system
    assert "{your algorithm description here}" in system and "Return a value." in system
    assert "### REFERENCE IMPLEMENTATION" in init and "return 9.0" in init
    assert "No additional domain knowledge provided." in init
    planner = users(llm, "planner")[0]
    assert "## Generation 0 Population Statistics:" in planner and "- **op3**: " + OPERATOR_DESCRIPTIONS["op3"] in planner
    first, later = users(llm, "generate")[4], users(llm, "generate")[6]
    assert "You are tasked is to " + OPERATOR_DESCRIPTIONS["op3"] in first
    assert "PARENT ALGORITHMS (1 total)" in first and re.search(r"Description: \{Return 4\.9[67]\.\}", first)
    assert "No successful experience retrieved." in first and "No failed experience retrieved." in first
    assert "LESSON" in later  # The second generation retrieves the first generation's distilled lessons.
    reflection = users(llm, "reflect")[0]
    assert "successfully improved performance" in reflection and '"best_parent_objective": 4.9' in reflection
    summary = read_json(tmp_path / "summary.json")
    assert summary["status"] == "finished" and summary["budget_used"] == 8 == len(task.evaluated)
    assert summary["refineevo"]["operator_usage"] == {"op3": 2}


def test_population_keeps_the_best_unique_objectives(tmp_path):
    values = {1: 3.0, 2: 3.0, 3: 2.0, 4: 7.0}
    instance, _ = method(tmp_path, Scripted(values=lambda n: values.get(n, 1.0)), budget=4)
    instance.run()
    assert instance._phase == "evolve"
    assert [i["objective"] for i in instance._population] == [2.0, 3.0]


def test_unparsed_responses_spend_budget_and_retry_up_to_five_times(tmp_path):
    llm = Scripted(generator=lambda n: "no code" if n <= 5 else program(5.0 - n))
    instance, task = method(tmp_path, llm, budget=7)
    instance.run()
    rejected = events(tmp_path, "sample_rejected")
    assert [e["status"] for e in rejected] == ["parse_failed"] * 5
    # The first slot gives up after five replies; the next slot evaluates its first reply.
    assert len(task.evaluated) == 2 and read_json(tmp_path / "summary.json")["budget_used"] == 7
    assert [r["operator"] for r in candidates(tmp_path)] == ["init", "init"]


def test_duplicates_of_population_code_spend_budget_and_retry_three_times(tmp_path):
    instance, task = method(tmp_path, Scripted(), budget=4)
    instance.run()
    parent_code = instance._population[0]["code"]
    llm = Scripted(generator=lambda n: f"{{Same.}}\n```python\n{parent_code}```")
    instance._llm, instance._max_sample_nums = llm, 8
    slot = instance._get_offspring(instance._population, "op3")
    assert [e["status"] for e in events(tmp_path, "sample_rejected")] == ["duplicate"] * 3
    assert slot[4]["other_inf"]["duplicate_attempts"] == 3 and instance._tot_sample_nums == 8
    assert len(task.evaluated) == 5


def test_transport_failure_does_not_spend_budget(tmp_path):
    instance, task = method(tmp_path, Scripted(fail_first=True), budget=4)
    instance.run()
    assert read_json(tmp_path / "summary.json")["budget_used"] == 4 and len(task.evaluated) == 4
    # The failed call skips its initial slot, as upstream's batch would lose that thread.
    assert instance._init_index == 4 and len(instance._population) == 2


def test_malformed_planner_reply_takes_the_heuristic_rule(tmp_path):
    llm = Scripted(planner=lambda n: '{"strategy": "exploitation"}')
    instance, _ = method(tmp_path, llm, budget=8)
    instance.run()
    strategy = events(tmp_path, "strategy")
    assert strategy[0]["fallback"] == "unparsed_reply" and strategy[0]["operators"] == ["op1", "op2"]
    # op1 and op2 select two parents.
    assert "PARENT ALGORITHMS (2 total)" in users(llm, "generate")[4]


def test_five_exploitation_rounds_force_exploration(tmp_path):
    instance, _ = method(tmp_path, Scripted(), budget=4)
    instance.run()
    planner = instance._planner
    for generation in range(5):
        planner.history["decisions"].append({"decision": {"strategy": "exploitation"}})
    assert planner.decide_strategy(instance._population, 5)["selected_operators"] == ["op1", "op2"]


def test_stagnant_operator_is_rewritten_and_old_experiences_stop_matching(tmp_path):
    llm = Scripted(check=lambda n: '```json\n{"needs_improvement": true, "reasoning": "flat", "confidence": 0.9, '
                                   '"improvement_urgency": "high"}\n```')
    # 4 initial + 3 generations of 2 offspring give op3 three history records; generation 4 refines op3.
    instance, _ = method(tmp_path, llm, budget=12)
    instance.run()
    assert llm.counts["check"] == 1 and llm.counts["improve"] == 1
    assert instance._op_desc["op3"] == instance._planner.operator_descriptions["op3"] == "IMPROVED1"
    improve = users(llm, "improve")[0]
    assert "## TASK: Improve Operator op3" in improve and "Gen 1: Mean=" in improve
    last = users(llm, "generate")[-1]
    assert "You are tasked is to IMPROVED1" in last
    # Stored experiences carry the old operator goal, so retrieval under the new goal finds none.
    assert "No successful experience retrieved." in last and "No failed experience retrieved." in last
    refined = events(tmp_path, "operator_refinement")
    assert refined[0]["status"] == "improved" and refined[0]["trigger"] == "flat"


def test_memory_scores_follow_the_batch_start_snapshot(tmp_path):
    instance, _ = method(tmp_path, Scripted(), budget=4)
    instance.run()
    pool = instance._pool
    pool.add([{"embedding": [1.0, 0.0, 0.0], "document": "q",
               "metadata": {"query": "q", "operator_goal": "g", "is_success": True, "distilled_items": "[]",
                            "n_retrieves": 0, "score": 0}}])
    first, _ = pool.retrieve_memories("q", "g")
    second, _ = pool.retrieve_memories("q", "g")
    assert pool.data[0]["metadata"]["n_retrieves"] == 2
    assert pool.update_memory_score(first[0], True) == 1 and pool.data[0]["metadata"]["score"] == 1
    # Both slots read score 0; as with upstream threads, the later write replaces the earlier one.
    assert pool.update_memory_score(second[0], False) == -1 and pool.data == []


def test_slots_of_one_batch_do_not_see_each_others_experience(tmp_path):
    llm = Scripted()
    instance, _ = method(tmp_path, llm, budget=6, pop_size=2)
    instance._max_sample_nums = 7  # Leave budget so that the batch's reflections run.
    instance.run()
    second_slot = users(llm, "generate")[5]
    assert "No successful experience retrieved." in second_slot
    assert len(events(tmp_path, "experience_added")) == 2


def test_all_invalid_initial_population_ends_the_search(tmp_path):
    instance, task = method(tmp_path, Scripted(values=lambda n: -1.0), budget=20)
    instance.run()
    summary = read_json(tmp_path / "summary.json")
    # The search ends as complete; no valid program exists for final selection.
    assert summary["status"] == "no_valid_root" and summary["budget_used"] == 4
    assert summary["termination"] == "No valid individual in the initial population."


def test_missing_entrypoint_spends_budget_without_evaluation(tmp_path):
    llm = Scripted(generator=lambda n: program(1.0, name="other") if n == 1 else program(5.0 - n))
    instance, task = method(tmp_path, llm, budget=4)
    instance.run()
    assert [e["status"] for e in events(tmp_path, "sample_rejected")] == ["invalid_program"]
    assert len(task.evaluated) == 3 and read_json(tmp_path / "summary.json")["budget_used"] == 4


def test_embedding_failure_stops_at_the_last_checkpoint(tmp_path):
    instance, _ = method(tmp_path, Scripted(), budget=20, embedding=Embedding(fail_at=1))
    instance.run()
    summary = read_json(tmp_path / "summary.json")
    assert summary["status"] == "aborted" and summary["budget_used"] == 4
    assert events(tmp_path, "embedding_call")[0]["status"] == "error"
    resumed, task = method(tmp_path, Scripted(), budget=20)
    report = recover(resumed, tmp_path)
    # The last checkpoint follows the fourth initial slot; resuming forms the population without new calls.
    assert report["boundary"] == "initialization" and resumed._init_index == 4 and task.evaluated == []


def comparable(instance):
    return ([(i["candidate_id"], i["code"], i["objective"]) for i in instance._population],
            instance._op_desc, instance._planner.history, [m["_id"] for m in instance._pool.data])


def test_resume_restores_the_generation_boundary_without_reevaluating(tmp_path):
    # 4 initial + generation 1 (planner, 2 offspring, 2 reflections), then stop inside generation 2.
    first, _ = method(tmp_path, Scripted(stop_after=11), budget=14)
    first.run()
    assert read_json(tmp_path / "summary.json")["status"] == "interrupted"
    # New values, so that no reply repeats a population member's code.
    resumed, task = method(tmp_path, Scripted(values=lambda n: 3.0 - 0.01 * n), budget=14)
    report = recover(resumed, tmp_path)
    assert report["recovery"] == "search_checkpoint" and report["boundary"] == "generation"
    assert task.evaluated == [] and resumed._phase == "evolve" and resumed._generation == 1
    assert len(resumed._pool.data) == 2
    used = report["budget_used"]
    resumed.run()
    assert len(task.evaluated) == 14 - used
    ids = [r["candidate_id"] for r in candidates(tmp_path)]
    assert ids == list(range(1, len(ids) + 1))
    assert read_json(tmp_path / "summary.json")["budget_used"] == 14


def test_resume_during_initialization_continues_with_the_next_slot(tmp_path):
    first, _ = method(tmp_path, Scripted(stop_after=2), budget=10)
    first.run()
    resumed, task = method(tmp_path, Scripted(), budget=10)
    report = recover(resumed, tmp_path)
    assert report["boundary"] == "initialization" and resumed._init_index == 2
    assert len(resumed._init_population) == 2
    resumed.run()
    assert len(task.evaluated) == 10 - 2


@pytest.mark.parametrize("task_name", [
    "tsp_construct", "cvrp_aco", "fssp_gls", "graph_colouring", "jssp_construct", "op_aco"])
def test_real_solver_entry_final_selection_monitor_and_config_rebuild(tmp_path, monkeypatch, task_name):
    from benchmarks.tasks import training_task
    from traceaad.instance_evaluation import _instance_view
    from experiments.refineevo import run
    from experiments.infra import base
    from experiments.infra.resume_baseline import build, close_recovery_method
    from experiments.visualization import ResultsMonitor

    monkeypatch.delenv("TRACEAAD_SCHEDULER_SOCKET", raising=False)
    monkeypatch.setenv("TRACEAAD_EVALUATION_LOG", "")
    original, settings = training_task(task_name)
    task = _instance_view(original, 0, original.timeout_seconds)
    monkeypatch.setattr(base, "training_task", lambda *a, **k: (task, settings))
    template = str(task.template_program)
    reply = "{Keep the template.}\n```python\n" + template.strip() + "\n```"
    clients = []

    def client(**kwargs):
        llm = TokenLLM(reply, reply)
        clients.append(llm)
        return llm

    monkeypatch.setattr(run, "build_llm_client", client)
    spec = run.RunSpec(task=task_name, budget=2, eval_workers=1, run_name="integration", experiments_root=tmp_path)
    directory = run.run_experiment(spec)
    summary = read_json(directory / "summary.json")
    assert summary["status"] == "finished" and summary["selection_evaluations"] == 0
    found = candidates(directory)
    assert [r["operator"] for r in found] == ["init", "init"] and all(r["valid"] for r in found)
    assert found[0]["fitness"] == found[1]["fitness"] == summary["best"]["fitness"]
    assert parse_program((directory / "best_program.py").read_text()) is not None
    if task_name in ("tsp_construct", "cvrp_aco"):
        assert "- Try " in clients[0].calls[0][0][1]["content"]  # Upstream external knowledge.
    detail = ResultsMonitor(tmp_path).run_detail("refineevo", task_name, "integration")
    assert detail["best"]["code"] == (directory / "best_program.py").read_text()
    rebuilt = build(read_json(directory / "run_config.json"), directory)
    try:
        assert clients[-1].calls == []
        report = recover(rebuilt, directory)
        assert report["boundary"] == "final_selection_only" and rebuilt._tot_sample_nums == 2
    finally:
        close_recovery_method(rebuilt)


def test_planner_summary_counts_operators():
    planner = Planner(lambda *a: None, OPERATOR_DESCRIPTIONS)
    planner.history["decisions"] = [{"decision": {"strategy": "exploration", "selected_operators": ["op1", "op2"]}},
                                    {"decision": {"strategy": "exploitation", "selected_operators": ["op3"]}}]
    assert planner.get_summary()["operator_usage"] == {"op1": 1, "op2": 1, "op3": 1}
    assert re.match(r"create a new algorithm", planner.operator_descriptions["op1"])
