"""A failed generation must not erase ReEvo's usable parents or spent budget."""

import copy
from unittest.mock import Mock

import pytest

from baselines.reevo import ReEvo, ReEvoProfiler
from experiments.infra.resume_baseline import recover
from tests.support import TinyEvaluation, TokenLLM
from traceaad.storage import append_jsonl, rows, write_json


@pytest.fixture
def make_reevo(tmp_path, monkeypatch):
    monkeypatch.setattr(ReEvo, "_short_term_reflection", lambda *args: "Prefer valid programs.")
    monkeypatch.setattr(ReEvo, "_long_term_reflection", lambda *args: "Prefer valid programs.")
    methods = []

    def create(values, budget=9):
        llm = TokenLLM()
        llm.draw_sample = Mock(side_effect=[
            f"```python\ndef score(x):\n    return {value}\n```" for value in values
        ])
        method = ReEvo(llm, TinyEvaluation(), ReEvoProfiler(tmp_path),
                       max_sample_nums=budget, pop_size=2, init_pop_size=2)
        methods.append(method)
        return method

    yield create
    for method in methods:
        method._evaluation_executor.shutdown()


def initialize(method):
    seed = copy.deepcopy(method._function_to_evolve)
    seed.body = "    return 1\n"
    method._seed_function = method._evaluate_function(seed, "seed")
    method._initialize_population()


@pytest.mark.parametrize("budget", [5, 6, 9])
def test_all_invalid_offspring_keep_parents_and_budget(make_reevo, tmp_path, budget):
    method = make_reevo([2, 3, *["1 / 0"] * 3, 4, 5, 6], budget)
    initialize(method)
    parents = [str(f) for f in method._population.population]
    method._run_evolution_generation()
    assert [str(f) for f in method._population.population] == parents
    assert method._tot_sample_nums == min(budget, 6)
    assert method._elite_function.score == -3
    events = list(rows(tmp_path / "events.jsonl"))
    assert sum(e.get("kind") == "candidate" for e in events) == method._tot_sample_nums
    assert all(e["members"] for e in events if e.get("kind") == "population")
    assert any(e.get("data", {}).get("event") == "population_fallback" for e in events)
    if budget == 9:
        method._run_evolution_generation()
        assert method._tot_sample_nums == 9
        assert method._elite_function.score == -6
        assert not set(parents).intersection(str(f) for f in method._population.population)


def test_valid_mutation_replaces_parents_after_failed_crossovers(make_reevo):
    method = make_reevo([2, 3, "1 / 0", "1 / 0", 4], budget=6)
    initialize(method)
    method._run_evolution_generation()
    assert [f.score for f in method._population.population] == [-4]
    assert method._tot_sample_nums == 6


def test_recovery_skips_legacy_empty_population_without_reevaluation(make_reevo, tmp_path):
    write_json(tmp_path / "run_config.json", {
        "method": "reevo", "task": "toy", "budget": 9,
        "seed": 0, "backend": "test", "run_name": "test",
    })
    first = make_reevo([2, 3, *["1 / 0"] * 3])
    initialize(first)
    first._run_evolution_generation()
    # Older ReEvo versions committed an empty population before stopping.
    generation = first._population.generation + 1
    append_jsonl(tmp_path / "events.jsonl", {
        "kind": "population", "generation": generation, "members": [],
    })
    write_json(tmp_path / "resume.json", {"method_state": {
        "generation": generation, "population_size": 0, "sample_count": 6,
    }})
    resumed = make_reevo([4, 5, 6])
    report = recover(resumed, tmp_path)
    assert report["budget_used"] == 6 and report["remaining"] == 3
    assert resumed._population.generation == generation
    assert len(resumed._population) == 2
    assert resumed._sampler.llm.draw_sample.call_count == 0
    resumed._run_evolution_generation()
    assert resumed._tot_sample_nums == 9 and resumed._elite_function.score == -6
