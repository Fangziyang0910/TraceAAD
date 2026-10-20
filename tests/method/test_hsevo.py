"""Offline checks for the HSEvo loop, its budget accounting, harmony search and recovery."""

import re

import numpy as np
import pytest

from baselines.hsevo import HSEvo
from baselines.hsevo.hsevo import SCIENTISTS, Individual
from baselines.hsevo.sampler import code_to_function, extract_to_hs, parameter_ranges
from baselines.shared.profiler import ProfilerBase
from core import Evaluation
from core.code import parse_program
from experiments.infra.resume_baseline import recover
from tests.support import TokenLLM
from traceaad.storage import read_json, rows, write_json


TEMPLATE = 'def solve(x: float) -> float:\n    """Return a value."""\n    return 9.0\n'


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


def hs_reply(signature="def solve_v2(x: float, alpha: float = 0.5) -> float:", ranges="{'alpha': (0.0, 1.0)}"):
    return (f"```python\n{signature}\n    return alpha\n```\n"
            f"```python\nparameter_ranges = {ranges}\n```")


class Scripted(TokenLLM):
    """Answer each HSEvo prompt by its role; generated values follow `values`."""

    def __init__(self, values=None, *, hs=None, stop_after=None, fail_first=False):
        super().__init__()
        self.values = values or (lambda n: 5.0 - 0.01 * n)
        self.hs, self.stop_after, self.fail_first = hs or (lambda n: hs_reply()), stop_after, fail_first
        self.generated = 0
        self.stages = []

    def draw_sample_with_details(self, prompt, **kwargs):
        if self.stop_after is not None and len(self.calls) >= self.stop_after:
            raise KeyboardInterrupt
        self.calls.append(prompt)
        if self.fail_first and len(self.calls) == 1:
            raise ValueError("Offline transport failure")
        system, user = prompt[0]["content"], prompt[1]["content"]
        if system.startswith("You are an expert in code review"):
            self.stages.append("hs")
            content = self.hs(len(self.stages))
        elif "### List heuristics" in user:
            self.stages.append("flash")
            content = f"**Analysis:** ANALYSIS{len(self.calls)}\n**Experience:** EXPERIENCE{len(self.calls)}"
        elif "### Current self-reflection" in user:
            self.stages.append("comprehensive")
            content = f"- Keywords: COMPREHENSIVE{len(self.calls)}"
        else:
            self.stages.append("generate")
            self.generated += 1
            content = f"```python\ndef solve_v2(x: float) -> float:\n    return {self.values(self.generated)}\n```"
        return {"content": content, "model": "offline",
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}


def method(directory, llm, *, budget=40, **options):
    task = ToyTask()
    write_json(directory / "run_config.json", {"method": "hsevo", "task": "toy", "budget": budget,
                                                "method_params": {}})
    np.random.seed(3)
    options = {"pop_size": 2, "init_pop_size": 3, **options}
    return HSEvo(llm, task, ProfilerBase(directory), max_sample_nums=budget, **options), task


def candidates(directory):
    return [r for r in rows(directory / "events.jsonl") if r["kind"] == "candidate"]


def events(directory):
    return [r["data"] for r in rows(directory / "events.jsonl") if r["kind"] == "method"]


def test_prompts_follow_upstream_roles_and_order(tmp_path):
    llm = Scripted()
    # Seed, 3 initial, 2 crossover, 1 mutation, 5 + 5 harmony-search variants.
    instance, task = method(tmp_path, llm, budget=17)
    instance.run()
    assert llm.stages == ["generate"] * 3 + ["flash", "comprehensive"] + ["generate"] * 3 + ["hs"]
    for i in range(3):
        system, user = llm.calls[i][0]["content"], llm.calls[i][1]["content"]
        assert system.startswith(SCIENTISTS[i]) and user.startswith(SCIENTISTS[i])
        assert "def solve_v1(x: float) -> float:" in user and "Be very creative and give `solve_v2`" in user
    flash = llm.calls[3][1]["content"]
    assert "[Heuristics 1st]" in flash and "ranked from best to worst" in flash
    crossover = llm.calls[5][1]["content"]
    assert "### Better code\ndef solve_v0(x: float) -> float:" in crossover
    assert "- ANALYSIS4" in crossover and "COMPREHENSIVE5" in crossover
    mutation = llm.calls[7][1]["content"]
    assert "think outside the box" in mutation and "COMPREHENSIVE5" in mutation
    assert "import numpy as np" in llm.calls[8][1]["content"]
    summary = read_json(tmp_path / "summary.json")
    assert summary["status"] == "finished" and summary["budget_used"] == 17 == len(task.evaluated)
    assert [r["operator"] for r in candidates(tmp_path)].count("harmony_search") == 10


def test_harmony_search_tunes_the_extracted_parameters_and_keeps_only_the_best(tmp_path):
    instance, _ = method(tmp_path, Scripted(), budget=17)
    instance.run()
    tuned = [r for r in candidates(tmp_path) if r["operator"] == "harmony_search"]
    assert all(r["program"]["parent_id"] is not None for r in tuned)
    hs = [i for i in instance._population if i.operator == "harmony_search"]
    assert len(hs) == 1 and hs[0].tryHS and hs[0].obj == min(r["fitness"] for r in tuned if r["valid"])
    assert "alpha: float = 0." in hs[0].code
    event = next(e for e in events(tmp_path) if e["event"] == "harmony_search")
    assert event["parameters"] == ["alpha"] and event["hs_best"] <= event["init_best"]


def test_failed_harmony_search_spends_its_evaluations_and_retries(tmp_path):
    # Every tuned variant raises, so upstream would refund five evaluations per try.
    llm = Scripted(hs=lambda n: hs_reply(ranges="{'alpha': (-2.0, -1.0)}"))
    instance, task = method(tmp_path, llm, budget=25)
    instance.run()
    rejected = [e for e in events(tmp_path) if e["event"] == "harmony_search_rejected"]
    assert [e["status"] for e in rejected] == ["all_invalid"] * 3 and llm.stages.count("hs") == 3
    assert read_json(tmp_path / "summary.json")["budget_used"] == 25 == len(task.evaluated)


def test_unparsed_response_spends_budget_without_evaluation(tmp_path):
    llm = Scripted(values=lambda n: "(" if n == 1 else 5.0 - n)
    instance, task = method(tmp_path, llm, budget=4)
    instance.run()
    assert [e["status"] for e in events(tmp_path) if e["event"] == "sample_rejected"] == ["parse_failed"]
    assert len(task.evaluated) == 3 and read_json(tmp_path / "summary.json")["budget_used"] == 4
    assert [i.exec_success for i in instance._population] == [False, True, True]


def test_transport_failure_does_not_spend_budget(tmp_path):
    llm = Scripted(fail_first=True)
    instance, task = method(tmp_path, llm, budget=4)
    instance.run()
    assert read_json(tmp_path / "summary.json")["budget_used"] == 4 and len(task.evaluated) == 4
    # The failed call loses its scientist, as a smaller upstream batch would.
    assert [r["operator"] for r in candidates(tmp_path)] == ["seed", "init", "init", "crossover"]
    assert llm.calls[1][0]["content"].startswith(SCIENTISTS[1])


def test_all_tied_parents_use_unranked_prompts(tmp_path):
    llm = Scripted(values=lambda n: 4.0)
    instance, _ = method(tmp_path, llm, budget=8)
    instance.run()
    assert any(e["event"] == "selection_fallback" for e in events(tmp_path))
    flash = next(c[1]["content"] for c in llm.calls if "### List heuristics" in c[1]["content"])
    assert "perform equally well" in flash and "[Heuristics 1]" in flash and "1st" not in flash
    crossover = next(c[1]["content"] for c in llm.calls if "COMBINING" in c[1]["content"])
    assert "### Code 1" in crossover and "Better code" not in crossover


def test_all_invalid_population_ends_the_search_as_upstream(tmp_path):
    instance, task = method(tmp_path, Scripted(values=lambda n: -1.0), budget=20)
    instance.run()
    summary = read_json(tmp_path / "summary.json")
    assert summary["status"] == "finished" and summary["budget_used"] == 4
    assert summary["termination"] == "All individuals are invalid."


def test_harmony_selection_keeps_the_upstream_index(tmp_path):
    instance, _ = method(tmp_path, Scripted())
    first, second, third = (Individual("a", "x", obj=3.0), Individual("b", "x", obj=1.0),
                            Individual("c", "x", obj=2.0))
    instance._population = [first, second, third]
    assert instance._sel_individual_hs() is second
    # Candidates are [first, third]; upstream then marks population[1] again.
    assert instance._sel_individual_hs() is second and not third.tryHS


def test_parameter_extraction_matches_upstream_and_rejects_what_upstream_cannot_use():
    ranges, block = extract_to_hs(hs_reply())
    assert ranges == {"alpha": (0.0, 1.0)}
    assert block.splitlines()[0] == "def solve_v2(x: float, alpha: float = {alpha}) -> float:"
    assert extract_to_hs(hs_reply(ranges="{'alpha': (0, np.inf)}")) == (None, None)
    assert extract_to_hs(hs_reply(ranges="{'alpha': (0.0, 1.0, 2.0)}")) == (None, None)
    assert extract_to_hs("```python\ndef f():\n    return 1\n```") == (None, None)
    assert parameter_ranges("import numpy as np\nparameter_ranges = {'a': (0, np.pi)}")["a"][1] == pytest.approx(np.pi)
    assert parameter_ranges("while True:\n    pass", timeout=1) is None


def test_entrypoint_lookup_follows_upstream_name_order():
    code = "def solve_v3(x):\n    return 3\n\ndef solve_v1(x):\n    return 1\n"
    function = code_to_function(code, "solve")
    assert function is not None and function.name == "solve" and "return 1" in str(function)
    assert code_to_function("def other(x):\n    return 1\n", "solve") is None


def comparable(population):
    return [(i.candidate_id, i.code, i.obj, i.tryHS) for i in population]


def test_resume_restores_the_generation_boundary_without_reevaluating(tmp_path):
    first, _ = method(tmp_path, Scripted(stop_after=10), budget=30)
    first.run()
    checkpoint = read_json(tmp_path / "summary.json")
    assert checkpoint["status"] == "interrupted"
    resumed, task = method(tmp_path, Scripted(), budget=30)
    report = recover(resumed, tmp_path)
    assert report["recovery"] == "search_checkpoint" and report["boundary"] == "generation"
    assert task.evaluated == [] and resumed._phase == "evolve"
    assert comparable(resumed._population) == comparable(first._population)
    assert resumed._elitist in resumed._population or resumed._elitist.operator == "seed"
    used = report["budget_used"]
    resumed.run()
    assert len(task.evaluated) == 30 - used
    ids = [r["candidate_id"] for r in candidates(tmp_path)]
    assert ids == list(range(1, len(ids) + 1))
    assert read_json(tmp_path / "summary.json")["budget_used"] == 30


def test_resume_during_initialization_continues_with_the_next_scientist(tmp_path):
    first, _ = method(tmp_path, Scripted(stop_after=2), budget=10, init_pop_size=4)
    first.run()
    llm = Scripted()
    resumed, task = method(tmp_path, llm, budget=10, init_pop_size=4)
    report = recover(resumed, tmp_path)
    assert report["boundary"] == "initialization" and resumed._init_index == 2
    resumed.run()
    assert llm.calls[0][0]["content"].startswith(SCIENTISTS[2])
    assert len(task.evaluated) == 10 - 3


@pytest.mark.parametrize("task_name", [
    "tsp_construct", "cvrp_aco", "fssp_gls", "graph_colouring", "jssp_construct", "op_aco"])
def test_real_solver_entry_final_selection_monitor_and_config_rebuild(tmp_path, monkeypatch, task_name):
    from benchmarks.tasks import training_task
    from traceaad.instance_evaluation import _instance_view
    from experiments.hsevo import run
    from experiments.infra import base
    from experiments.infra.resume_baseline import build, close_recovery_method
    from experiments.visualization import ResultsMonitor

    monkeypatch.delenv("TRACEAAD_SCHEDULER_SOCKET", raising=False)
    monkeypatch.setenv("TRACEAAD_EVALUATION_LOG", "")
    original, settings = training_task(task_name)
    task = _instance_view(original, 0, original.timeout_seconds)
    monkeypatch.setattr(base, "training_task", lambda *a, **k: (task, settings))
    template = str(task.template_program)
    name = parse_program(template).functions[0].name
    reply = "```python\n" + re.sub(rf"\b{name}\b", name + "_v2", template) + "\n```"
    clients = []

    def client(**kwargs):
        llm = TokenLLM(reply)
        clients.append(llm)
        return llm

    monkeypatch.setattr(run, "build_llm_client", client)
    spec = run.RunSpec(task=task_name, budget=2, eval_workers=1, run_name="integration", experiments_root=tmp_path)
    directory = run.run_experiment(spec)
    summary = read_json(directory / "summary.json")
    assert summary["status"] == "finished" and summary["selection_evaluations"] == 0
    found = candidates(directory)
    assert [r["operator"] for r in found] == ["seed", "init"] and all(r["valid"] for r in found)
    assert found[0]["fitness"] == found[1]["fitness"] == summary["best"]["fitness"]
    assert parse_program((directory / "best_program.py").read_text()) is not None
    detail = ResultsMonitor(tmp_path).run_detail("hsevo", task_name, "integration")
    assert detail["best"]["code"] == (directory / "best_program.py").read_text()
    rebuilt = build(read_json(directory / "run_config.json"), directory)
    try:
        assert clients[-1].calls == []
        report = recover(rebuilt, directory)
        assert report["boundary"] == "final_selection_only" and rebuilt._tot_sample_nums == 2
    finally:
        close_recovery_method(rebuilt)
