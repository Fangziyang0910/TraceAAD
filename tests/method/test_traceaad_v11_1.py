import json
from pathlib import Path

import pytest

from core import Evaluation
from traceaad.v11_1 import TraceAADV111
from traceaad.v11_1.prompts import OPERATOR_INSTRUCTIONS
from traceaad.v11_1.selection import (
    OPERATORS, PARENT_TEMPERATURE, reference_pool, sample_references, score_nodes,
)
from traceaad.v11_1.storage import read_journal
from traceaad.v11_1.tree import Node


class TinyEvaluation(Evaluation):
    def __init__(self):
        super().__init__(template_program="def score(x):\n    pass",
                         task_description="Return a numeric score.", safe_evaluate=False)

    def evaluate_program(self, program_str, callable_func, **kwargs):
        return callable_func(1)


class FakeLLM:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def count_tokens(self, text):
        return len(text.split())

    def count_prompt_tokens(self, text):
        return self.count_tokens(text) + 7

    def draw_sample_with_details(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        response = next(self.responses)
        return {"content": response, "finish_reason": "stop", "usage": {}}


def response(value):
    return (
        "Idea: Return the requested constant.\n"
        "Code:\n```python\n"
        f"def score(x):\n    return {value}\n"
        "```"
    )


def make_method(path, llm=None, **kwargs):
    return TraceAADV111(
        evaluation=TinyEvaluation(), llm=llm or FakeLLM(), run_dir=path,
        budget=kwargs.pop("budget", 1000), n_roots=kwargs.pop("n_roots", 1), **kwargs,
    )


def add_node(method, fitness, *, parent_id=None, idea="idea", code=None, operator="Init"):
    return method.tree.add(
        code=code or f"def score(x):\n    return {fitness}", idea=idea,
        fitness=fitness, evaluation_id=len(method.tree.nodes) + 1,
        parent_id=parent_id, operator=operator,
    )


def test_parent_scores_are_node_level_and_use_fixed_temperature(tmp_path):
    method = make_method(tmp_path, budget=1)
    first = add_node(method, 1, code="def score(x):\n    return 1")
    second = add_node(method, 1, code="def score(x):\n    return 1")
    second.attempts = 4

    nodes, percentiles, scores = score_nodes(method.tree.all_nodes())

    assert [node.id for node in nodes] == [first.id, second.id]
    assert percentiles == pytest.approx([0.5, 0.5])
    low_attempts_bonuses = [score - percentile for score, percentile in zip(scores, percentiles)]
    assert low_attempts_bonuses[0] > low_attempts_bonuses[1] and scores[0] > scores[1]
    assert method.mechanism["parent_temperature"] == PARENT_TEMPERATURE == 0.2


def test_reference_pool_keeps_duplicate_code_nodes_and_excludes_only_parent(tmp_path):
    method = make_method(tmp_path, budget=1)
    parent = add_node(method, 3, idea="parent")
    duplicate_a = add_node(method, 2, idea="a", code=parent.code)
    duplicate_b = add_node(method, 1, idea="b", code=parent.code)

    pool = reference_pool(method.tree.all_nodes(), parent)
    assert [node.id for node in pool] == [duplicate_a.id, duplicate_b.id]


def test_rank_softmax_references_are_without_replacement():
    nodes = [Node(i, f"def score(x):\n    return {i}", f"idea {i}", 4 - i)
             for i in range(4)]
    sampled = sample_references(nodes, 8, __import__("random").Random(0))
    assert len(sampled) == 4
    assert len({node.id for node in sampled}) == 4


def test_fuse_expands_first_sampled_reference_and_records_donor(tmp_path, monkeypatch):
    llm = FakeLLM(response(3))
    method = make_method(tmp_path, llm, budget=1)
    parent = add_node(method, 3, idea="current")
    donor = add_node(method, 2, parent_id=parent.id, idea="donor", operator="Refine",
                     code="def score(x):\n    return 20")
    real_choices = method.rng.choices

    def fuse_only(population, weights=None, **kwargs):
        if tuple(population) == OPERATORS:
            return ["Fuse"]
        return real_choices(population, weights=weights, **kwargs)

    monkeypatch.setattr(method.rng, "choices", fuse_only)

    pending = method._schedule_candidate()

    assert pending.requested_operator == "Fuse"
    assert pending.operator == "Fuse"
    assert pending.donor_id == donor.id
    assert pending.reference_ids
    assert "# Reference Nodes" in pending.prompt
    assert "# Reference Algorithm" in pending.prompt
    assert donor.code in pending.prompt
    assert parent.attempts == 1


def test_repairs_do_not_increment_attempts_again(tmp_path):
    llm = FakeLLM(response(1), "bad output", response(3))
    method = make_method(tmp_path, llm, budget=2)
    method.run()

    events = read_journal(method.storage.events_path)
    assert [event["status"] for event in events] == ["ok", "invalid_output", "ok"]
    assert events[1]["parent_selected"] is True
    assert events[2]["parent_selected"] is False
    assert method.tree.parent_selections == 1
    assert method.tree.nodes[0].attempts == 1
    assert method.tree.nodes[1].attempts == 0
    state = json.loads((Path(tmp_path) / "tree_state.json").read_text())
    assert {n["id"]: n["attempts"] for n in state["nodes"]} == {0: 1, 1: 0}


def test_attempts_and_method_identity_survive_resume(tmp_path):
    method = make_method(tmp_path, FakeLLM(response(1), response(2)), budget=2)
    method.run()
    state = json.loads((Path(tmp_path) / "tree_state.json").read_text())
    assert state["mechanism"]["method"] == "v111"
    assert {n["id"]: n["attempts"] for n in state["nodes"]} == {0: 1, 1: 0}

    resumed = make_method(tmp_path, FakeLLM(), budget=2)
    resumed.run()
    assert {node.id: node.attempts for node in resumed.tree.all_nodes()} == {0: 1, 1: 0}
    assert resumed.llm.calls == []


def test_interrupted_candidate_is_redone_from_checkpoint(tmp_path, monkeypatch):
    method = make_method(tmp_path, FakeLLM(response(1), response(2)), budget=2)
    original = method._generate

    def interrupt_on_second_candidate(candidate):
        if candidate.candidate_id == 2:
            raise KeyboardInterrupt
        return original(candidate)

    monkeypatch.setattr(method, "_generate", interrupt_on_second_candidate)
    with pytest.raises(KeyboardInterrupt):
        method.run()

    # The in-flight candidate is discarded; the second run redoes it from the
    # checkpoint (same RNG state, so the same parent is selected again).
    resumed = make_method(tmp_path, FakeLLM(response(2)), budget=2)
    resumed.run()
    assert resumed.tree.nodes[0].attempts == 1
    assert resumed.tree.parent_selections == 1
    events = read_journal(resumed.storage.events_path)
    assert [event["status"] for event in events] == ["ok", "ok"]


def test_operator_instructions_keep_pivot_and_fuse_distinct():
    assert "different main decision mechanism" in OPERATOR_INSTRUCTIONS["Pivot"]
    assert "Do not mechanically concatenate programs" in OPERATOR_INSTRUCTIONS["Fuse"]
