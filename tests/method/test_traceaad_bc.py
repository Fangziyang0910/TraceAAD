from pathlib import Path

from llm4ad.base import Evaluation
from traceaad.bc import (
    TraceAADV10BudgetV11Context,
    TraceAADV11BudgetV10Context,
)


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
    model = "fake"
    base_url = "http://fake"
    temperature = 0.0
    top_p = 1.0
    enable_thinking = False

    def count_tokens(self, text):
        return len(text.split())

    def count_prompt_tokens(self, text):
        return len(text.split())


def node(method, code, fitness, idea, parent_id=None, operator="Init"):
    return method.tree.add(
        code=code,
        idea=idea,
        fitness=fitness,
        evaluation_id=len(method.tree.nodes) + 1,
        parent_id=parent_id,
        operator=operator,
    )


def make_method(cls, tmp_path, **kwargs):
    return cls(
        evaluation=TinyEvaluation(),
        llm=FakeLLM(),
        run_dir=Path(tmp_path),
        budget=10,
        n_roots=1,
        traj_gens=8,
        output_tokens=8192,
        max_input_tokens=24576,
        seed=0,
        **kwargs,
    )


def test_b_keeps_v11_scheduler_and_v10_generic_context(tmp_path):
    method = make_method(TraceAADV11BudgetV10Context, tmp_path)
    parent = node(method, "def score(x):\n    return 5", 5, "parent")
    donor = node(method, "def score(x):\n    return 4", 4, "donor")
    method.codebook.register(parent)
    method.codebook.register(donor)

    prompt = method.builder.build(parent, "Fuse", donor)

    assert method.METHOD == "bc_v11budget_v10ctx"
    assert method.mechanism["scheduler"] == "v11_percentile_plus_bonus"
    assert method.mechanism["context"] == "v1011_generic_trajectory_donor"
    assert "# Reference Algorithm" in prompt
    assert "aims to outperform both" in prompt
    assert "# Reference Nodes" not in prompt


def test_c_keeps_v10_scheduler_and_v11_reference_context(tmp_path):
    method = make_method(TraceAADV10BudgetV11Context, tmp_path)
    parent = node(method, "def score(x):\n    return 5", 5, "parent")
    reference = node(method, "def score(x):\n    return 4", 4, "reference")

    prompt, retained = method.reference_builder.build(
        parent, "Fuse", [reference], require_minimal=True
    )

    assert method.METHOD == "bc_v10budget_v11ctx"
    assert method.mechanism["scheduler"] == "v1011_quality_ess_plus_pivot_uniform"
    assert method.mechanism["context"] == "v110_trajectory_reference"
    assert [item.id for item in retained] == [reference.id]
    assert "# Reference Nodes" in prompt
    assert "reference ideas" in prompt
    assert "# Reference Algorithm" not in prompt
    assert "aims to improve performance on the task" in prompt


def test_c_reference_pool_deduplicates_codes_and_excludes_parent(tmp_path):
    method = make_method(TraceAADV10BudgetV11Context, tmp_path)
    parent = node(method, "def score(x):\n    return 5", 5, "parent")
    node(method, "def score(x):\n    return 5 + 0", 4, "same behavior")
    newer = node(method, "def score(x):\n    return 4", 3, "newer")
    node(method, "def score(x):\n    return 4", 4, "latest")

    pool = method.reference_pool(parent)

    assert {item.id for item in pool} == {1, newer.id + 1}
