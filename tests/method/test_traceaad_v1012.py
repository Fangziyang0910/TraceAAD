"""Tests for TraceAAD V10.12: short formation trajectory plus dual archive profile cards."""

import random
from pathlib import Path

from llm4ad.base import Evaluation
from llm4ad.method.traceaad_v10_12 import TraceAADV1012
from llm4ad.method.traceaad_v10_12.prompts import TrajectoryBuilder
from llm4ad.method.traceaad_v10_12.selection import rank_softmax_sample
from llm4ad.method.traceaad_v10_12.storage import read_journal


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
        if isinstance(response, BaseException):
            raise response
        return response if isinstance(response, dict) else {
            "content": response, "finish_reason": "stop", "usage": {"completion_tokens": 20},
        }


def response(value=1):
    return (
        "Idea: Return the requested constant.\n"
        "Code:\n"
        "```python\n"
        f"def score(x):\n    return {value}\n"
        "```"
    )


class NumberedNode:
    def __init__(self, id, fitness, idea="test idea", parent_id=None, code="def score(x):\n    return 1"):
        self.id = id
        self.fitness = fitness
        self.idea = idea
        self.parent_id = parent_id
        self.code = code
        self.operator = "Init" if parent_id is None else "Refine"


def test_rank_softmax_sample_properties():
    nodes = [NumberedNode(i, i * 1.5) for i in range(20)]
    rng = random.Random(42)
    sample = rank_softmax_sample(nodes, 2, rng, tau=8.0)
    assert len(sample) == 2
    assert sample[0].id != sample[1].id

    # Determinism
    sample2 = rank_softmax_sample(nodes, 2, random.Random(42), tau=8.0)
    assert [n.id for n in sample] == [n.id for n in sample2]

    # Few nodes (N < k)
    one_node = [NumberedNode(99, 10.0)]
    assert [n.id for n in rank_softmax_sample(one_node, 2, rng)] == [99]
    assert rank_softmax_sample([], 2, rng) == []


def test_trajectory_builder_with_both_history_and_profile_cards():
    llm = FakeLLM()
    contract = "# Task\nReturn score."
    nodes = {
        0: NumberedNode(0, 1.0, "root idea"),
        1: NumberedNode(1, 2.0, "step 1 idea", parent_id=0),
        2: NumberedNode(2, 3.0, "step 2 idea", parent_id=1),
        3: NumberedNode(3, 8.0, "archive card A"),
        4: NumberedNode(4, 9.5, "archive card B"),
    }
    builder = TrajectoryBuilder(
        llm, contract, max_tokens=10000, max_events=8,
        lookup=nodes.get, all_nodes=lambda: list(nodes.values())
    )

    parent = nodes[2]
    references = [nodes[3], nodes[4]]
    prompt = builder.build(parent, "Refine", donor=None, references=references)

    # Must contain current algorithm
    assert "# Current Algorithm" in prompt
    assert "Fitness: 3.0" in prompt

    # Must contain design history (Step 1, Step 2)
    assert "# Design History of the Current Algorithm" in prompt
    assert "Step 1 | Refine | Fitness: 1.0 -> 2.0" in prompt
    assert "Step 2 | Refine | Fitness: 2.0 -> 3.0" in prompt

    # Must contain profile cards, ordered best-to-worst
    assert "# Archive Profile Cards" in prompt
    assert "2 independently evaluated algorithm(s)" in prompt
    assert "Profile Card 1 | Fitness: 9.5" in prompt
    assert "Idea: archive card B" in prompt
    assert "Profile Card 2 | Fitness: 8.0" in prompt
    assert "Idea: archive card A" in prompt

    # Profile Card 1 should appear before Profile Card 2
    pos1 = prompt.find("Profile Card 1 | Fitness: 9.5")
    pos2 = prompt.find("Profile Card 2 | Fitness: 8.0")
    assert pos1 < pos2

    # Operator instruction must be present
    assert "# Design Task" in prompt
    assert "# Output" in prompt


def test_trajectory_builder_empty_references():
    llm = FakeLLM()
    contract = "# Task\nReturn score."
    root = NumberedNode(0, 1.0, "root idea")
    builder = TrajectoryBuilder(
        llm, contract, max_tokens=10000, max_events=8,
        lookup={0: root}.get, all_nodes=lambda: [root]
    )
    prompt = builder.build(root, "Refine", donor=None, references=[])
    assert "# Archive Profile Cards" not in prompt
    assert "# Current Algorithm" in prompt


def test_traceaad_v1012_execution_and_reference_tracking(tmp_path):
    # 2 roots + 3 evolutionary steps = 5 responses
    responses = [response(1), response(2), response(3), response(4), response(5)]
    llm = FakeLLM(*responses)
    engine = TraceAADV1012(
        evaluation=TinyEvaluation(),
        llm=llm,
        run_dir=tmp_path,
        budget=5,
        n_roots=2,
        traj_gens=8,
        n_profile_cards=2,
        seed=1,
    )
    engine.run()

    events = read_journal(tmp_path / "events.jsonl")
    assert len(events) == 5

    # Events 1 and 2 are Init (roots): reference_ids should be empty
    assert events[0]["operator"] == "Init"
    assert events[0]["reference_ids"] == []
    assert events[1]["operator"] == "Init"
    assert events[1]["reference_ids"] == []

    # Events 3, 4, 5 are evolutionary operators:
    # They should sample profile cards from the archive (excluding parent)
    for event in events[2:]:
        assert event["operator"] in ("Refine", "Tune", "Pivot", "Fuse")
        assert len(event["reference_ids"]) > 0
        assert len(event["reference_ids"]) <= 2
        # Parent must not be in its own reference cards
        assert event["parent_id"] not in event["reference_ids"]

    # Mechanism dictionary check
    assert engine.mechanism["method"] == "v1012"
    assert engine.mechanism["n_profile_cards"] == 2
    assert engine.mechanism["profile_card_tau"] == 8.0

