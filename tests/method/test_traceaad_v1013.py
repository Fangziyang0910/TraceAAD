import json
import random
from pathlib import Path

from core import Evaluation
from traceaad.v10_13 import TraceAADV1013
from traceaad.v10_13.parsing import parse_candidate, template_target
from traceaad.v10_13.prompts import OPERATOR_INSTRUCTIONS, PromptBuilder
from traceaad.v10_13.selection import (
    PIVOT_UNIFORM_PROBABILITY,
    quality_distribution,
    sample_parent,
    select_donor,
)
from traceaad.v10_13.storage import read_journal
from traceaad.v10_13.tree import Node


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
        return {"content": next(self.responses), "finish_reason": "stop", "usage": {}}


def response(value, heading="Idea:"):
    return (
        f"{heading} Try a direct decision rule that returns {value}.\n"
        "Code:\n```python\n"
        f"def score(x):\n    return {value}\n"
        "```"
    )


def make_method(path, llm=None, **kwargs):
    return TraceAADV1013(
        evaluation=TinyEvaluation(), llm=llm or FakeLLM(), run_dir=path,
        budget=kwargs.pop("budget", 1000), n_roots=kwargs.pop("n_roots", 1), **kwargs,
    )


def add_node(method, fitness, *, parent_id=None, idea="idea", code=None):
    code = code or f"def score(x):\n    return {fitness}"
    return method.tree.add(
        code=code, idea=idea, fitness=fitness,
        evaluation_id=len(method.tree.nodes) + 1, parent_id=parent_id,
        operator="Init",
    )


def test_quality_allocator_uses_v1010_ess_and_pivot_exploration():
    nodes = [Node(index, f"def score(x):\n    return {index}", "idea", index)
             for index in range(12)]
    probabilities, stats = quality_distribution(nodes)
    assert abs(stats["quality_ess"] - 8.0) < 1e-6
    _, selection = sample_parent(nodes, "Pivot", random.Random(0))
    assert selection["parent_ess"] > stats["quality_ess"]
    assert selection["parent_probability"] >= PIVOT_UNIFORM_PROBABILITY / len(nodes)


def test_donor_selection_is_one_archive_program_without_exact_copy():
    nodes = [
        Node(0, "def score(x):\n    return x", "a", 1),
        Node(1, "def score(x):\n    return x", "copy", 9),
        Node(2, "def score(x):\n    return x + 1", "b", 2),
    ]
    donor, stats = select_donor(nodes, nodes[0], random.Random(2))
    assert donor.id == 2
    assert stats["donor_pool_size"] == 1


def test_prompts_are_short_and_operator_specific_without_behavior_checklist(tmp_path):
    method = make_method(tmp_path, budget=1)
    parent = add_node(method, 3)
    donor = add_node(method, 4, code="def score(x):\n    return x + 1")
    builder = PromptBuilder(
        method.llm, method.task_contract, max_tokens=20000, history_depth=3,
        lookup=method.tree.nodes.get, all_nodes=method.tree.all_nodes,
    )
    text = builder.build(parent, "Fuse", donor)
    assert "Host Algorithm" in text and "Donor Algorithm" in text
    assert "Idea: idea" in text
    assert "Do not mechanically concatenate" in text
    assert "Change" not in text and "Evidence" not in text and "Preserve" not in text
    assert "candidate ranking" not in text
    assert "promising improvement" in OPERATOR_INSTRUCTIONS["Refine"]
    assert "influential parameters" in OPERATOR_INSTRUCTIONS["Tune"]
    history = builder.build(parent, "Refine")
    assert "Recent Design History" not in history  # root has no parent edge


def test_parser_accepts_simple_markdown_idea_heading():
    interface, _ = template_target("def score(x):\n    pass")
    parsed, error = parse_candidate(response(4, "**Idea:**"), "stop", interface,
                                    "def score(x):\n    pass")
    assert error is None
    assert parsed.idea.startswith("Try a direct")


def test_repair_does_not_allocate_a_second_parent_opportunity(tmp_path):
    llm = FakeLLM(response(1), "bad output", response(3))
    method = make_method(tmp_path, llm, budget=2)
    method.run()
    events = read_journal(method.storage.events_path)
    assert [event["status"] for event in events] == ["ok", "invalid_output", "ok"]
    assert events[1]["parent_selected"] is True
    assert events[2]["parent_selected"] is False
    assert method.tree.parent_selections == 1


def test_checkpoint_contains_only_evolutionary_state_and_resumes(tmp_path):
    method = make_method(tmp_path, FakeLLM(response(1), response(2)), budget=2)
    method.run()
    state = json.loads((Path(tmp_path) / "tree_state.json").read_text())
    assert state["mechanism"]["method"] == "v1013"
    assert "route_stats" not in state
    assert all("attempts" in node for node in state["nodes"])

    resumed = make_method(tmp_path, FakeLLM(), budget=2)
    resumed.run()
    assert resumed.llm.calls == []
    assert resumed.tree.parent_selections == method.tree.parent_selections


def test_operator_instructions_are_algorithmic_moves():
    assert "promising improvement" in OPERATOR_INSTRUCTIONS["Refine"]
    assert "Calibrate influential parameters" in OPERATOR_INSTRUCTIONS["Tune"]
    assert "different main decision idea" in OPERATOR_INSTRUCTIONS["Pivot"]
    assert "adapt ideas" in OPERATOR_INSTRUCTIONS["Fuse"]
