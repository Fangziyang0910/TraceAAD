import ast
import json
from pathlib import Path

import pytest

from llm4ad.base import Evaluation
from llm4ad.base.evaluate import EvaluationOutcome
from llm4ad.method.traceaad_v10_11 import TraceAADV1011
from llm4ad.method.traceaad_v10_11.traceaad import mix_uniform
from llm4ad.method.traceaad_v10_11.core import UnknownEvaluation, read_journal
from llm4ad.method.traceaad_v10_11.errors import repair_prompt, template_target
from llm4ad.method.traceaad_v10_11.trajectory import (
    INIT_REFERENCE_INSTRUCTION,
    OPERATOR_INSTRUCTIONS,
    OUTPUT,
)

MODULE_ROOT = Path(__file__).resolve().parents[2] / "llm4ad" / "method" / "traceaad_v10_11"


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


def method(path, llm=None, **kwargs):
    return TraceAADV1011(
        evaluation=TinyEvaluation(),
        llm=llm or FakeLLM(),
        run_dir=path,
        **{"budget": 1000, "n_roots": 1, **kwargs},
    )


def add(tree, fitness, parent=None, code="def score(x):\n    return 1", operator=None):
    return tree.add(
        code=code,
        idea=f"idea {len(tree.nodes)}",
        fitness=fitness,
        evaluation_id=len(tree.nodes) + 1,
        parent_id=parent,
        operator="Init" if parent is None else (operator or "Refine"),
    )


def test_v1011_does_not_import_historical_methods():
    forbidden = []
    for path in sorted(MODULE_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.startswith("llm4ad.method.") and not name.startswith(
                        "llm4ad.method.traceaad_v10_11"):
                    forbidden.append(f"{path.name}: {name}")
                if "traceaad_v" in name and "traceaad_v10_11" not in name:
                    forbidden.append(f"{path.name}: {name}")
    assert forbidden == []


def test_v1011_template_target_is_local():
    interface, stub = template_target(
        "import math\n\n"
        "def score(x: int) -> float:\n"
        '    """Return a numeric score."""\n'
        "    return math.sqrt(x)\n"
    )
    assert interface[0] == "score"
    assert interface[1] == "x: int"
    assert stub == (
        "def score(x: int) -> float:\n"
        '    """Return a numeric score."""\n'
        "    pass"
    )
    with pytest.raises(ValueError, match="exactly one function"):
        template_target("x = 1")


def test_v1011_requires_one_idea(tmp_path):
    m = method(tmp_path, budget=1)
    assert m.parse_response("```python\ndef score(x):\n    return 1\n```") is None
    parsed = m.parse_response(
        "Idea: constant score\nCode:\n```python\ndef score(x):\n    return 1\n```"
    )
    assert parsed[0] == "constant score"
    assert ast.parse(parsed[1]).body[0].name == "score"


def test_v1011_does_not_inject_evaluator_design_notes(tmp_path):
    evaluation = TinyEvaluation()
    evaluation.design_notes = "internal evaluator implementation detail"
    m = TraceAADV1011(evaluation=evaluation, llm=FakeLLM(),
                      run_dir=tmp_path, budget=1, n_roots=1)
    assert "internal evaluator implementation detail" not in m.task_contract
    assert "Evaluator Semantics" not in m.task_contract
    assert "Interface Notes" not in m.task_contract


def test_v1011_context_exposes_only_target_function_stub(tmp_path):
    m = method(tmp_path, budget=1)
    assert "# Task" in m.task_contract
    assert "Target function:" in m.task_contract
    assert "def score(x):" in m.task_contract
    assert "Public template:" not in m.task_contract
    assert "Evaluator" not in m.task_contract


def test_v1011_runs_function_through_template(tmp_path):
    m = method(tmp_path, FakeLLM(response(7)), budget=1)
    m.run()
    assert m.tree.best().fitness == 7
    assert "def score" in m.tree.best().code


def test_v1011_accepts_dependencies_but_requires_the_declared_target(tmp_path):
    m = method(tmp_path, budget=1)
    parsed = m.parse_response(
        "Idea: valid summary\nCode:\n```python\nimport math\n\n"
        "def helper(x):\n    return math.floor(x) + 1\n\n"
        "def score(x):\n    return helper(x)\n```"
    )
    assert parsed is not None
    assert "import math" in parsed[2]
    assert "def helper" in parsed[2]
    assert "def score" in parsed[2]
    assert m.parse_response(
        "Idea: missing target\nCode:\n```python\ndef helper(x):\n    return x\n```"
    ) is None
    long_idea = "word " * 1700
    assert m.parse_response(
        f"Idea: {long_idea}\nCode:\n```python\ndef score(x):\n    return 1\n```"
    ) is None
    assert m.parse_response(
        "Idea: valid summary\nCode:\n```python\ndef score(x):\n    return 1\n```\nExtra explanation"
    ) is None


def test_v1011_allows_a_concise_idea_below_the_token_limit(tmp_path):
    m = method(tmp_path, budget=1)
    idea = " ".join(["mechanism"] * 180)
    parsed = m.parse_response(
        f"Idea: {idea}\nCode:\n```python\ndef score(x):\n    return 1\n```"
    )
    assert parsed[0] == idea


def test_v1011_repair_prompt_includes_evaluator_error_details():
    prompt = repair_prompt(
        "Task contract",
        "Idea: broken\nCode:\n```python\ndef score(x):\n    return x\n```",
        {"error_type": "ValueError", "error": "array dimensions do not match",
         "reason": "runtime_error"},
    )
    assert "ValueError: array dimensions do not match" in prompt
    assert "Keep the Idea to no more than 200 words." in prompt


def test_v1011_returns_target_function_and_keeps_small_comments(tmp_path):
    m = method(tmp_path, budget=1)
    parsed = m.parse_response(
        "Idea: Add a constant offset.\nCode:\n```python\n"
        "def score(x):\n    # one useful comment\n    return x + 1\n```"
    )
    assert parsed[1].startswith("def score")
    assert not parsed[1].lstrip().startswith("#")


def test_v1011_history_preserves_the_full_bounded_idea(tmp_path):
    m = method(tmp_path, budget=1)
    root = add(m.tree, 1)
    child = add(m.tree, 2, root.id, code="def score(x):\n    return x",
                operator="Refine")
    child.idea = " ".join(["long"] * 200)
    prompt = m.builder.build(child, "Refine")
    assert prompt.count("long") == 200


def test_v1011_history_code_is_explicitly_opt_in(tmp_path):
    m = method(tmp_path, budget=1, history_code=True)
    root = add(m.tree, 1)
    middle = add(m.tree, 2, root.id, code="def score(x):\n    return x + 1")
    child = add(m.tree, 3, middle.id, code="def score(x):\n    return x + 2")
    default = method(tmp_path / "default", budget=1)
    default.tree = m.tree
    default.builder.lookup = m.tree.nodes.get
    prompt = m.builder.build(child, "Refine")
    assert "Step 1 | Refine" in prompt
    assert "Step 2 | Refine" in prompt
    assert prompt.count("```python\ndef score(x):\n    return x + 1\n```") == 1
    assert prompt.count("```python\ndef score(x):\n    return x + 2\n```") == 2
    default_history = default.builder.build(child, "Refine").split("# Design Task", 1)[0]
    assert "Code:" not in default_history
    assert m.mechanism["history_code"] is True


def test_v1011_zero_history_omits_only_the_trajectory_context(tmp_path):
    baseline = method(tmp_path / "baseline")
    ablation = method(tmp_path / "ablation", traj_gens=0)
    for m in (baseline, ablation):
        root = add(m.tree, 1)
        add(m.tree, 2, root.id, code="def score(x):\n    return x")
        add(m.tree, 3, code="def score(x):\n    return x + 1")
    assert ablation.mechanism["traj_gens"] == 0
    assert ablation.builder.build_initial() == baseline.builder.build_initial()
    for operator in ("Refine", "Tune", "Pivot", "Fuse"):
        def prompt(m):
            return m.builder.build(m.tree.nodes[1], operator,
                                   m.tree.nodes[2] if operator == "Fuse" else None)
        expected = "\n\n\n".join(part for part in prompt(baseline).split("\n\n\n")
                                 if not part.startswith("# Design History"))
        assert prompt(ablation) == expected


def test_v1011_history_truncates_earliest_steps_when_prompt_exceeds_budget(tmp_path, capsys):
    m = method(tmp_path, budget=1, history_code=True)
    root = add(m.tree, 1, code="def score(x):\n    return x")
    middle = add(m.tree, 2, root.id, code="def score(x):\n    return x + 1")
    child = add(m.tree, 3, middle.id, code="def score(x):\n    return x + 2")
    assert m.mechanism["history_code"] is True

    one_step = method(tmp_path / "one_step", budget=1, history_code=True, traj_gens=1)
    one_step.tree = m.tree
    one_step.builder.lookup = m.tree.nodes.get
    expected_prompt = one_step.builder.build(child, "Refine")
    assert expected_prompt.count("Step ") == 1

    m.builder.max_tokens = m.builder.count(expected_prompt)
    truncated = m.builder.build(child, "Refine")
    assert truncated == expected_prompt
    assert truncated.count("return x + 1") == 0
    assert "formation history truncated to 1 of 2 steps" in capsys.readouterr().out


def test_v1011_history_truncation_still_raises_when_no_history_fits(tmp_path):
    m = method(tmp_path, budget=1, history_code=True)
    root = add(m.tree, 1, code="def score(x):\n    return x")
    child = add(m.tree, 2, root.id, code="def score(x):\n    return x + 1")

    zero = method(tmp_path / "zero", budget=1, history_code=True, traj_gens=0)
    zero.tree = m.tree
    zero.builder.lookup = m.tree.nodes.get
    bare = zero.builder.build(child, "Refine")

    m.builder.max_tokens = m.builder.count(bare) - 1
    with pytest.raises(ValueError, match="context budget"):
        m.builder.build(child, "Refine")


def test_v1011_generation_uses_fixed_output_budget(tmp_path):
    llm = FakeLLM(response())
    m = method(tmp_path, llm, budget=1)
    prompt = "token " * 16773
    m.pending = {
        "candidate_id": 1,
        "operator": "Refine",
        "prompt": prompt,
        "prompt_tokens": 16773,
        "prompt_hash": "test",
        "llm_attempts": 0,
    }
    m._generate_pending()
    assert llm.calls[0][1]["max_tokens"] == 8192


def test_v1011_prompt_policy_uses_compact_generic_design_instructions(tmp_path):
    m = method(tmp_path, budget=1)
    first_init = m.builder.build_initial()
    assert "Fitness: higher is better." in first_init
    assert "# Search Context" not in first_init
    assert OPERATOR_INSTRUCTIONS["Init"] in first_init
    assert INIT_REFERENCE_INSTRUCTION not in first_init

    root = add(m.tree, 1)
    child = add(m.tree, 2, root.id, code="def score(x):\n    return x")
    add(m.tree, 3, code="def score(x):\n    return x + 1")
    assert INIT_REFERENCE_INSTRUCTION in m.builder.build_initial()

    fuse = m.builder.build(child, "Fuse", donor=m.tree.nodes[2])
    assert "# Current Algorithm" in fuse
    assert "# Reference Algorithm" in fuse
    assert "# Host Algorithm" not in fuse
    assert "# Donor Algorithm" not in fuse
    assert "# Design History of the Current Algorithm" in fuse
    assert "Use the recorded design changes and their results to guide this design." in fuse
    assert "host-centered" not in fuse


def test_v1011_prompt_policy_and_output_contract_are_recorded(tmp_path):
    m = method(tmp_path, budget=1)
    assert m.mechanism["prompt_policy"] == "generic_design_v1"
    assert "describing how the algorithm computes its output" in OUTPUT
    assert "main change from any supplied algorithms" in OUTPUT
    assert "including the target function and any supporting code" in OUTPUT


def test_runtime_failure_repairs_once_and_charges_every_evaluation(tmp_path):
    llm = FakeLLM(response(1), response("missing_name"), response(3))
    m = method(tmp_path, llm, budget=3)
    m.run()
    events = read_journal(m.events_path)
    assert [event["status"] for event in events] == ["ok", "eval_failed", "ok"]
    assert [event["evaluation_id"] for event in events] == [1, 2, 3]
    assert events[1]["error_type"] == "NameError" and "missing_name" in events[1]["error"]
    assert events[1]["traceback"] and "NameError" in events[1]["traceback"]
    assert events[2]["repair_of"] == 2 and events[2]["parent_id"] == events[1]["parent_id"]
    assert m.parent_selection_counts == {0: 1}
    assert "missing_name" in llm.calls[2][0]
    assert "intended decision method" in llm.calls[2][0]
    assert [call["stage"] for call in read_journal(m.llm_calls_path)] == [
        "generation", "generation", "repair",
    ]


def test_failed_repair_returns_to_normal_search_and_parse_uses_no_eval(tmp_path):
    m = method(tmp_path, FakeLLM(response(1), "bad output", "still bad", response(4)), budget=2)
    m.run()
    events = read_journal(m.events_path)
    assert [event["status"] for event in events] == [
        "ok", "invalid_output", "invalid_output", "ok",
    ]
    assert events[2]["repair_of"] == 2 and "repair_of" not in events[3]
    assert sum(1 for e in events if e["status"] == "ok") == 2
    assert sum(m.parent_selection_counts.values()) == 2


def test_initial_failure_repair_and_final_budget_boundary(tmp_path):
    m = method(tmp_path / "initial", FakeLLM(response("1/0"), response(2)), budget=2)
    m.run()
    assert m.tree.best().parent_id is None and len(m.tree.roots) == 1
    assert read_journal(m.events_path)[1]["repair_of"] == 1
    final = method(tmp_path / "final", FakeLLM(response(1), response("1/0"), response(3)), budget=2)
    final.run()
    assert len(final.llm.calls) == 2 and final.budget_used == 2


def test_initialization_reaches_target_roots_then_enters_search(tmp_path):
    m = method(tmp_path, FakeLLM(response(1), response(2)), budget=2, n_roots=2)
    m.run()
    assert len(m.tree.roots) == 2 and m.budget_used == 2
    assert all(node.parent_id is None for node in m.tree.all_nodes())
    assert m._schedule()["operator"] != "Init"


def test_quality_and_pivot_distributions(tmp_path):
    base = method(tmp_path / "base")
    shifted = method(tmp_path / "shifted")
    p1, stats1 = base._quality_distribution([add(base.tree, fitness) for fitness in range(1, 101)])
    p2, stats2 = shifted._quality_distribution(
        [add(shifted.tree, 10 * fitness + 1000) for fitness in range(1, 101)]
    )
    assert p1 == pytest.approx(p2)
    assert stats1["quality_ess"] == pytest.approx(8) and stats2["quality_ess"] == pytest.approx(8)

    m = method(tmp_path / "pivot")
    nodes = [add(m.tree, fitness) for fitness in range(4)]
    quality, _ = m._quality_distribution(nodes)
    assert m.node_distribution(nodes, "Refine")[0] == pytest.approx(quality)
    assert m.node_distribution(nodes, "Pivot")[0] == pytest.approx(
        [0.5 * value + 0.5 / 4 for value in quality]
    )
    assert m.mechanism["quality_ess_target"] == 8
    assert m.mechanism["pivot_uniform_probability"] == 0.5


def test_mix_uniform_uses_remaining_quality_mass():
    assert mix_uniform([0.8, 0.2], 0.2) == pytest.approx([0.74, 0.26])


@pytest.mark.parametrize("kind", ["prepare_error", "evaluation_error"])
def test_infrastructure_failure_stops_without_llm_repair(tmp_path, monkeypatch, kind):
    m = method(tmp_path, FakeLLM(response(1), response(2), response(3)), budget=3)
    m._advance()
    if kind == "prepare_error":
        monkeypatch.setattr(
            m.secure, "evaluate_program_with_details",
            lambda _: EvaluationOutcome(
                result=None, failure_kind=kind, error_type="OSError", error="worker setup failed",
            ),
        )
    else:
        def fail(_):
            raise OSError("evaluation transport failed")
        monkeypatch.setattr(m.secure, "evaluate_program_with_details", fail)
    with pytest.raises(RuntimeError, match="evaluation infrastructure failed"):
        m.run()
    assert read_journal(m.events_path)[-1]["reason"] == kind
    assert len(m.llm.calls) == 2


def test_transport_failure_during_repair_resumes_the_same_request(tmp_path):
    m = method(tmp_path, FakeLLM(response(1), response("missing"), RuntimeError("offline")), budget=3)
    with pytest.raises(RuntimeError, match="offline"):
        m.run()
    restored = method(tmp_path, FakeLLM(response(3)), budget=3)
    restored.run()
    assert restored.llm.calls[0][0] == m.llm.calls[2][0]
    assert restored.parent_selection_counts == {0: 1}
    assert [call["call_id"] for call in read_journal(m.llm_calls_path)] == ["1:1", "2:1", "3:1", "3:2"]
    assert read_journal(m.events_path)[-1]["repair_of"] == 2


def test_persisted_response_survives_crash_before_evaluator(tmp_path, monkeypatch):
    m = method(tmp_path, FakeLLM(response(3)), budget=1)

    def crash(_parsed):
        raise KeyboardInterrupt
    monkeypatch.setattr(m, "_evaluate_pending", crash)
    with pytest.raises(KeyboardInterrupt):
        m.run()
    assert json.loads(m.pending_path.read_text())["phase"] == "responded"
    resumed = method(tmp_path, budget=1)
    resumed.run()
    assert resumed.tree.best().fitness == 3 and resumed.llm.calls == []


def test_unknown_evaluation_blocks_without_redrawing(tmp_path, monkeypatch):
    m = method(tmp_path, FakeLLM(response()), budget=1)

    def die(_):
        raise KeyboardInterrupt
    monkeypatch.setattr(m.secure, "evaluate_program_with_details", die)
    with pytest.raises(KeyboardInterrupt):
        m.run()
    resumed = method(tmp_path, budget=1)
    with pytest.raises(UnknownEvaluation):
        resumed.run()
    summary = json.loads(resumed.summary_path.read_text())
    assert summary["status"] == "blocked" and resumed.budget_used == 0
    assert resumed.llm.calls == []


def test_v1011_persistence_keeps_single_copies(tmp_path):
    m = method(tmp_path, FakeLLM(response(1), response(2)), budget=2)
    m.run()

    calls = read_journal(m.llm_calls_path)
    assert calls and all("prompt" not in call for call in calls)
    assert all(call.get("prompt_hash") and call.get("response") for call in calls)
    assert not (tmp_path / "evaluations.jsonl").exists()

    events = read_journal(m.events_path)
    assert [event["best_fitness"] for event in events] == [1, 2]

    nodes = read_journal(m.nodes_path)
    assert [node["id"] for node in nodes] == [0, 1]
    assert "nodes" not in json.loads((tmp_path / "tree_state.json").read_text())

    resumed = method(tmp_path, FakeLLM(), budget=2)
    resumed.run()
    assert len(resumed.tree.nodes) == 2 and resumed.budget_used == 2
    assert resumed.llm.calls == []
