import json
import random

from core import Evaluation
from traceaad.v10_13 import TraceAADV1013
from traceaad.v10_13.parsing import parse_candidate, template_target
from traceaad.v10_13.prompts import PromptBuilder
from traceaad.v10_13.selection import (
    PARENT_UNIFORM_PROBABILITY,
    ess,
    mix_uniform,
    quality_distribution,
    sample_parent,
    sample_reference,
)
from traceaad.v10_13.storage import read_journal
from traceaad.v10_13.tree import Node


class TinyEvaluation(Evaluation):
    def __init__(self):
        super().__init__(
            template_program="def score(x):\n    pass",
            task_description="Return a numeric score.",
            safe_evaluate=False,
        )

    def evaluate_program(self, program_str, callable_func, **kwargs):
        return callable_func(1)


class FakeLLM:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def count_prompt_tokens(self, text):
        return len(text.split())

    def draw_sample_with_details(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return {
            "content": next(self.responses),
            "finish_reason": "stop",
            "usage": {},
            "model": "test",
        }


def response(value):
    return f"Idea: return {value}\n```python\ndef score(x):\n    return {value}\n```"


def make_method(path, llm, *, budget=1, n_roots=1, seed=0):
    return TraceAADV1013(
        evaluation=TinyEvaluation(),
        llm=llm,
        run_dir=path,
        budget=budget,
        n_roots=n_roots,
        seed=seed,
    )


def test_parent_distribution_keeps_ess_and_uniform_exploration():
    nodes = [Node(i, f"def score(x):\n    return {i}", "", i) for i in range(12)]
    quality = quality_distribution(nodes)
    mixed = mix_uniform(quality, PARENT_UNIFORM_PROBABILITY)

    assert abs(ess(quality) - 8.0) < 1e-6
    assert ess(mixed) > ess(quality)
    assert sample_parent(nodes, random.Random(0)) in nodes


def test_reference_is_a_different_program():
    nodes = [
        Node(0, "def score(x):\n    return x", "", 1),
        Node(1, "def score(x):\n    return x", "", 9),
        Node(2, "def score(x):\n    return x + 1", "", 2),
    ]
    assert sample_reference(nodes, nodes[0], random.Random(2)).id == 2


def test_parser_accepts_python_block_and_preserves_template_helpers():
    template = "import math\n\ndef score(x):\n    pass"
    interface, _ = template_target(template)
    parsed, error = parse_candidate(response("math.sqrt(x)"), "stop", interface, template)

    assert error is None
    assert parsed.idea == "return math.sqrt(x)"
    assert "import math" in parsed.program_code
    assert "return math.sqrt(x)" in parsed.program_code


def test_parser_applies_one_simple_edit():
    template = "def score(x):\n    pass"
    interface, _ = template_target(template)
    parent = "def score(x):\n    return x"
    edit = json.dumps({
        "mode": "edit",
        "idea": "add one",
        "edits": [{"search": "return x", "replacement": "return x + 1"}],
    })

    parsed, error = parse_candidate(edit, "stop", interface, template, base_code=parent)

    assert error is None
    assert parsed.idea == "add one"
    assert "return x + 1" in parsed.program_code


def test_parser_rejects_wrong_target_signature():
    template = "def score(x):\n    pass"
    interface, _ = template_target(template)
    parsed, error = parse_candidate(
        "def score(x, y):\n    return x + y", "stop", interface, template,
    )

    assert parsed is None
    assert error.startswith("signature_error")


def test_operator_prompts_only_include_relevant_context(tmp_path):
    method = make_method(tmp_path, FakeLLM(response(1)))
    root = method.tree.add(
        code="def score(x):\n    return 1",
        idea="root",
        fitness=1,
        parent_id=None,
        operator="Init",
    )
    child = method.tree.add(
        code="def score(x):\n    return 2",
        idea="child",
        fitness=2,
        parent_id=root.id,
        operator="Refine",
    )
    reference = method.tree.add(
        code="def score(x):\n    return 3",
        idea="reference",
        fitness=3,
        parent_id=None,
        operator="Init",
    )
    builder = PromptBuilder(
        method.llm,
        method.task_prompt,
        max_tokens=20000,
        history_depth=3,
        lookup=method.tree.nodes.get,
        all_nodes=method.tree.all_nodes,
    )

    refine = builder.build_development(child, "Refine")
    fuse = builder.build_development(child, "Fuse", reference)
    pivot = builder.build_development(child, "Pivot", reference)

    assert "Recent Design History" in refine.prompt
    assert "Reference Algorithm" in fuse.prompt
    assert reference.code not in pivot.prompt


def test_initial_prompt_does_not_offer_parent_edit(tmp_path):
    method = make_method(tmp_path, FakeLLM(response(1)))
    prompt = method.prompts.build_initial().prompt

    assert '"mode":"edit"' not in prompt


def test_run_saves_results_with_a_small_state(tmp_path):
    method = make_method(
        tmp_path,
        FakeLLM(response(1), response(2), response(3)),
        budget=3,
    )
    method.run()

    state = json.loads(method.storage.state_path.read_text())
    summary = json.loads(method.storage.summary_path.read_text())
    events = read_journal(method.storage.events_path)

    assert state.keys() == {
        "started_at", "rng_state", "candidate_count", "budget_used", "invalid_streak",
    }
    assert state["budget_used"] == len(read_journal(method.storage.nodes_path)) == 3
    assert len(events) == 3
    assert summary["status"] == "finished"
    assert summary["best"]["fitness"] == 3


def test_invalid_output_does_not_spend_evaluation_budget(tmp_path):
    method = make_method(
        tmp_path,
        FakeLLM("This is not Python.", response(1)),
        budget=1,
    )
    method.run()

    events = read_journal(method.storage.events_path)
    assert [event["status"] for event in events] == ["invalid_output", "ok"]
    assert [event["budget_used"] for event in events] == [0, 1]


def test_failed_evaluation_spends_budget_and_search_continues(tmp_path):
    method = make_method(
        tmp_path,
        FakeLLM(response("1 / 0"), response(2)),
        budget=2,
    )
    method.run()

    events = read_journal(method.storage.events_path)
    assert [event["status"] for event in events] == ["eval_failed", "ok"]
    assert method.evaluations_used == 2


def test_resume_reads_saved_state_and_finishes_remaining_budget(tmp_path):
    make_method(tmp_path, FakeLLM(response(1)), budget=1).run()
    resumed = make_method(tmp_path, FakeLLM(response(2)), budget=2)
    resumed.run()

    assert resumed.evaluations_used == 2
    assert len(resumed.tree.nodes) == 2
    assert len(read_journal(resumed.storage.events_path)) == 2
