import json

import pytest

from evotrace_v1.initialization import initialize
from evotrace_v1 import Config, Task


class ScriptedModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def complete(self, prompt, **settings):
        self.requests.append((prompt, settings))
        response = next(self.replies)
        if isinstance(response, Exception):
            raise response
        return response


def reply(value=1, *, body=None):
    return json.dumps({
        "analysis": "Use a constant scoring rule.",
        "design": f"Return the constant {value}.",
        "code": "def solve(x):\n    " + (body or f"return {value}") + "\n",
    })


@pytest.fixture
def task():
    return Task(
        "Return a finite score; lower is better.",
        "def solve(x):\n    pass\n",
        lambda source, function: function(3),
        time_limit=3,
        example_code="def solve(x):\n    return -100\n",
        example_design="A sample algorithm, not an evaluated candidate.",
    )


def test_default_initialization_stops_at_five_valid_nodes_without_evolution(task, tmp_path):
    model = ScriptedModel([reply(2)] * 6)
    result = initialize(task, model, run_dir=tmp_path)

    assert result.stop_reason == "initialized"
    assert result.evaluations == result.model_calls == len(model.requests) == 5
    tree = result.population
    assert tree.root_id == 0 and 0 not in tree.nodes
    assert tree.latest_node_id == 5
    assert tree.direction_evaluations == {i: 1 for i in range(1, 6)}
    assert set(tree.nodes) == set(tree.edges) == {1, 2, 3, 4, 5}
    assert all(node.fitness == 2 for node in tree.nodes.values())
    assert all(edge.parent_id == 0 and edge.child_id == node_id
               and edge.operator is None and edge.reference_id is None
               for node_id, edge in tree.edges.items())
    assert all(not node.selection_experience for node in tree.nodes.values())
    assert all(not edge.variation_experience for edge in tree.edges.values())

    attempts = [json.loads(line) for line in (tmp_path / "attempts.jsonl").read_text().splitlines()]
    assert [row["attempt"]["operator"] for row in attempts] == ["Init"] * 5
    saved = json.loads((tmp_path / "initialization.json").read_text())
    assert saved["stop_reason"] == "initialized" and saved["evaluations"] == 5
    assert set(saved["population"]["nodes"]) == {"1", "2", "3", "4", "5"}


def test_initialization_records_context_repairs_and_failed_leaves(task, tmp_path):
    bad = reply(body="raise ValueError('invalid design')")
    answers = ["not json", bad, bad, reply(4), reply(2), reply(3), reply(1), reply(5)]
    config = Config(evaluation_budget=7)
    model = ScriptedModel(answers)
    result = initialize(task, model, config=config, run_dir=tmp_path)

    assert result.evaluations == 7
    assert result.model_calls == len(model.requests) == 8
    assert result.stop_reason == "initialized"
    assert "[Example Algorithm]" in model.requests[0][0]
    assert any("[Existing Algorithmic Ideas]" in prompt for prompt, _ in model.requests)
    assert any("[Failed Attempts]" in prompt for prompt, _ in model.requests)

    tree = result.population
    assert len(tree.nodes) == 6
    assert sum(node.fitness is not None for node in tree.nodes.values()) == 5
    assert tree.nodes[1].fitness is None and "invalid design" in tree.nodes[1].failure
    assert tree.nodes[2].fitness == 4  # Program repair updates the same individual.
    assert tree.direction_evaluations == {1: 1, 2: 2, 3: 1, 4: 1, 5: 1, 6: 1}
    assert sum(tree.direction_evaluations.values()) == result.evaluations
    assert all(edge.parent_id == tree.root_id for edge in tree.edges.values())
    attempts = [json.loads(line) for line in (tmp_path / "attempts.jsonl").read_text().splitlines()]
    assert attempts[0]["attempt"]["repair_used"]
    assert len(attempts[0]["attempt"]["evaluations"]) == 1  # No second repair after format repair.
    assert [entry["program_id"] for entry in attempts[1]["attempt"]["evaluations"]] == [2, 2]
    assert [entry["status"] for entry in attempts[1]["attempt"]["evaluations"]] == ["error", "success"]


@pytest.mark.parametrize("budget", [0, 3])
def test_budget_exhaustion_returns_an_explicit_partial_population(task, budget):
    model = ScriptedModel([reply()] * 5)
    result = initialize(task, model, config=Config(evaluation_budget=budget))
    assert result.stop_reason == "evaluation_budget_exhausted"
    assert result.evaluations == result.model_calls == len(result.population.nodes) == budget


def test_last_evaluation_does_not_spend_an_extra_call_on_repair(task):
    model = ScriptedModel([reply(body="return float('nan')")])
    result = initialize(task, model, config=Config(evaluation_budget=1))
    assert result.stop_reason == "evaluation_budget_exhausted"
    assert result.evaluations == result.model_calls == 1
    assert result.population.nodes[1].fitness is None
    assert result.population.nodes[1].failure


@pytest.mark.parametrize("responses, calls", [
    ([RuntimeError("model unavailable")] * 2, 2),
    (["not json"] * 4, 4),
])
def test_unevaluated_attempt_limit_returns_no_phantom_nodes(task, responses, calls):
    result = initialize(task, ScriptedModel(responses), config=Config(max_unevaluated=2))
    assert result.stop_reason == "consecutive_unevaluated_limit"
    assert result.evaluations == 0 and result.model_calls == calls
    assert not result.population.nodes and not result.population.edges


def test_custom_initial_size_stops_before_the_next_attempt(task):
    result = initialize(task, ScriptedModel([reply(1), reply(2)]), config=Config(initial_size=2))
    assert result.stop_reason == "initialized"
    assert len(result.population.nodes) == result.evaluations == 2


def test_invalid_size_and_existing_run_are_rejected_before_requests(task, tmp_path):
    model = ScriptedModel([])
    with pytest.raises(ValueError, match="initial_size"):
        initialize(task, model, config=Config(initial_size=0))
    config_path = tmp_path / "run_config.json"
    config_path.write_text("existing run")
    with pytest.raises(FileExistsError, match="new run directory"):
        initialize(task, model, run_dir=tmp_path)
    assert config_path.read_text() == "existing run" and model.requests == []
