import ast
import json
import math
import random
from pathlib import Path

import pytest

from llm4ad.base import Evaluation
from llm4ad.base.evaluate import EvaluationOutcome
from llm4ad.method.traceaad_v11_0 import TraceAADV110
from llm4ad.method.traceaad_v11_0.core import (
    Node,
    UnknownEvaluation,
    read_journal,
    truncate_torn_tail,
)
from llm4ad.method.traceaad_v11_0.errors import OUTPUT, repair_prompt, template_target
from llm4ad.method.traceaad_v11_0.traceaad import (
    EXPLORATION_C,
    code_key,
    quality_percentiles,
    reciprocal_rank_sample,
    reference_ranks,
)
from llm4ad.method.traceaad_v11_0.trajectory import (
    OPERATOR_INSTRUCTIONS,
    REFERENCE_INTRO,
)

MODULE_ROOT = Path(__file__).resolve().parents[2] / "llm4ad" / "method" / "traceaad_v11_0"


class TinyEvaluation(Evaluation):
    def __init__(self):
        super().__init__(template_program="def score(x):\n    pass",
                         task_description="Return a numeric score.", safe_evaluate=False)

    def evaluate_program(self, program_str, callable_func, **kwargs):
        return callable_func(1)


class FakeLLM:
    def __init__(self, *responses, base_url=None, model=None):
        self.responses = iter(responses)
        self.calls = []
        self.base_url = base_url
        self.model = model

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
    return TraceAADV110(
        evaluation=TinyEvaluation(),
        llm=llm or FakeLLM(),
        run_dir=path,
        **{"budget": 1000, "n_roots": 1, **kwargs},
    )


def add(m, fitness, parent=None, code="def score(x):\n    return 1", operator=None, idea=None):
    node = m.tree.add(
        code=code,
        idea=idea if idea is not None else f"idea {len(m.tree.nodes)}",
        fitness=fitness,
        evaluation_id=len(m.tree.nodes) + 1,
        parent_id=parent,
        operator="Init" if parent is None else (operator or "Refine"),
    )
    m.codebook.register(node)
    return node


def fake_node(identifier, fitness):
    return Node(id=identifier, code=f"def score(x):\n    return {identifier}",
                idea=f"idea {identifier}", fitness=fitness)


def test_v110_does_not_import_historical_methods():
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
                        "llm4ad.method.traceaad_v11_0"):
                    forbidden.append(f"{path.name}: {name}")
                if "traceaad_v" in name and "traceaad_v11_0" not in name:
                    forbidden.append(f"{path.name}: {name}")
    assert forbidden == []


def test_quality_percentiles_use_midrank_across_codes():
    assert quality_percentiles([1, 2, 2, 4]) == pytest.approx([0, 0.5, 0.5, 1])
    assert quality_percentiles([3, 3, 3]) == pytest.approx([0.5, 0.5, 0.5])
    assert quality_percentiles([7]) == [0.5]
    assert quality_percentiles([5, 9, 5, 5]) == pytest.approx([1 / 3, 1, 1 / 3, 1 / 3])


def test_reference_ranks_use_midrank_on_fitness():
    assert reference_ranks([10, 8, 8, 5]) == pytest.approx([1, 2.5, 2.5, 4])
    assert reference_ranks([4, 4]) == pytest.approx([1.5, 1.5])


def test_reciprocal_rank_sample_weights_and_replacement():
    nodes = [fake_node(0, 10), fake_node(1, 8), fake_node(2, 6), fake_node(3, 4)]
    counts = [0, 0, 0, 0]
    for seed in range(1200):
        drawn = reciprocal_rank_sample(nodes, 1, random.Random(seed))
        counts[drawn[0].id] += 1
    # weights 1/1 : 1/2 : 1/3 : 1/4 normalize to .48 / .24 / .16 / .12
    assert counts[0] / 1200 == pytest.approx(0.48, abs=0.05)
    assert counts[3] < counts[2] < counts[1] < counts[0]
    drawn = reciprocal_rank_sample(nodes, 8, random.Random(0))
    assert len(drawn) == 4 and len({node.id for node in drawn}) == 4


def test_reference_pool_dedups_by_code_and_filters(tmp_path):
    m = method(tmp_path, budget=1)
    parent = add(m, 5, code="def score(x):\n    return 5", idea="current")
    add(m, 4, code="def score(x):\n    return 4", idea="   ")
    recent = add(m, 4, code="def score(x):\n    return 4", idea="kept")
    add(m, 3, code="def score(x):\n    return 5", idea="self copy")
    other = add(m, 4, code="def score(x):\n    return 4 + 0", idea="other")
    pool = m.reference_pool(parent)
    assert {node.id for node in pool} == {recent.id, other.id}


def test_selection_breaks_score_ties_and_formation_ties_uniformly(tmp_path):
    m = method(tmp_path, budget=1)
    add(m, 5, code="def score(x):\n    return 5")
    add(m, 5, code="def score(x):\n    return 5 + 0")
    add(m, 5, code="def score(x):\n    return 5")  # duplicate formation node of code A
    parents, tie_sizes = set(), set()
    for seed in range(24):
        m.rng = random.Random(seed)
        parent, selection = m.select_parent()
        parents.add(parent.id)
        tie_sizes.add(selection["tie_group_size"])
    assert tie_sizes == {2}
    assert parents == {0, 1, 2}


def test_exploration_bonus_favors_underdeveloped_codes(tmp_path):
    m = method(tmp_path, budget=1)
    m.step_counter = 500
    fresh = add(m, 2, code="def score(x):\n    return 2")
    developed = add(m, 2, code="def score(x):\n    return 2 + 0")
    for _ in range(5):
        m.codebook.note_attempt(code_key(developed.code))
    keys, means, percentiles, bonuses, scores = m._score_codes()
    fresh_index = keys.index(code_key(fresh.code))
    assert percentiles == pytest.approx([0.5, 0.5])
    assert bonuses[fresh_index] == pytest.approx(EXPLORATION_C * math.sqrt(math.log(501)))
    m.rng = random.Random(0)
    parent, _ = m.select_parent()
    assert parent.id == fresh.id


def test_settlement_counts_every_attempt_once_per_outcome(tmp_path):
    llm = FakeLLM(response(1), "bad output", response(3), response("missing_name"), response(4))
    m = method(tmp_path, llm, budget=4)
    m.run()
    events = read_journal(m.events_path)
    assert [event["status"] for event in events] == ["ok", "invalid_output", "ok", "eval_failed", "ok"]
    assert [event["evaluation_id"] for event in events] == [1, None, 2, 3, 4]
    assert events[2]["repair_of"] == 2 and events[4]["repair_of"] == 4
    assert events[2]["parent_id"] == events[1]["parent_id"] == 0
    assert events[2]["reference_ids"] == [] and events[4]["reference_ids"] == []
    assert m.budget_used == 4 and m.step_counter == 4
    assert m.codebook.attempts(code_key(m.tree.nodes[0].code)) == 2  # candidates 2 and 3
    assert m.codebook.attempts(code_key(m.tree.nodes[1].code)) == 2  # candidates 4 and 5
    assert m.codebook.attempts(code_key(m.tree.nodes[2].code)) == 0
    assert m.parent_selection_counts == {0: 2, 1: 2}


def test_duplicate_generation_merges_into_existing_code_stats(tmp_path):
    m = method(tmp_path, FakeLLM(response(1), response(1)), budget=2)
    m.run()
    entry = m.codebook.entries[code_key(m.tree.nodes[0].code)]
    assert len(entry["scores"]) == 2 and entry["scores"] == [1, 1]
    assert entry["attempts"] == 1
    assert entry["node_ids"] == [0, 1]
    assert len(read_journal(m.nodes_path)) == 2  # duplicates are still evaluated and recorded


def test_reference_capacity_trims_worst_fitness_first(tmp_path, capsys):
    m = method(tmp_path, budget=1)
    parent = add(m, 5, code="def score(x):\n    return 5", idea="current")
    refs = [add(m, fitness, code=f"def score(x):\n    return {fitness}",
                idea="x " * (60 * (6 - fitness))) for fitness in (4, 3, 2)]
    text, retained = m.reference_builder.build(parent, "Pivot", refs)
    assert [node.fitness for node in retained] == [4, 3, 2]
    two_refs = m.reference_builder.build(parent, "Pivot", refs[:2])[0]
    m.reference_builder.max_tokens = m.reference_builder.count(two_refs)
    text, retained = m.reference_builder.build(parent, "Pivot", refs)
    assert [node.fitness for node in retained] == [4, 3]
    assert "Reference 3" not in text
    assert "reference context trimmed to 2 of 3" in capsys.readouterr().out
    minimal = m.reference_builder.build(parent, "Pivot", [])[0]
    m.reference_builder.max_tokens = m.reference_builder.count(minimal) - 1
    with pytest.raises(ValueError, match="context budget"):
        m.reference_builder.build(parent, "Pivot", refs)


def test_fuse_falls_back_to_refine_when_pool_is_empty(tmp_path, monkeypatch):
    llm = FakeLLM(response(1), response(2))
    m = method(tmp_path, llm, budget=2)
    monkeypatch.setattr(m.rng, "choices", lambda *arguments, **keywords: ["Fuse"])
    root = add(m, 1, idea="")  # empty idea keeps the reference pool empty
    add(m, 2, parent=root.id, code="def score(x):\n    return 2", idea="child idea")
    m.run()
    events = read_journal(m.events_path)
    search = events[0]
    assert search["requested_operator"] == "Fuse" and search["operator"] == "Refine"
    assert search["selection"]["fallback_reason"] == "reference_pool_empty"
    assert search["reference_ids"] == []
    prompt = llm.calls[0][0]
    assert "# Design History of the Current Algorithm" in prompt
    assert OPERATOR_INSTRUCTIONS["Refine"] in prompt
    assert "# Reference Nodes" not in prompt


def test_fuse_falls_back_to_refine_when_references_are_trimmed_to_zero(tmp_path, monkeypatch):
    llm = FakeLLM(response(1), response(2))
    m = method(tmp_path, llm, budget=2)
    monkeypatch.setattr(m.rng, "choices", lambda *arguments, **keywords: ["Fuse"])
    long_idea = " ".join(["word"] * 400)
    root = add(m, 1, idea=long_idea)
    child = add(m, 2, parent=root.id, code="def score(x):\n    return 2",
                idea=" ".join(["child"] * 60))
    capacity = m.builder.count(m.builder.build(child, "Refine"))
    m.max_input_tokens = capacity
    m.builder.max_tokens = capacity
    m.reference_builder.max_tokens = capacity
    m.run()
    events = read_journal(m.events_path)
    assert events[0]["operator"] == "Refine"
    assert events[0]["selection"]["fallback_reason"] == "references_trimmed_to_zero"
    assert events[0]["reference_ids"] == []
    assert "# Reference Nodes" not in llm.calls[0][0]
    assert "# Design History" in llm.calls[0][0]


def test_pivot_never_falls_back(tmp_path, monkeypatch):
    llm = FakeLLM(response(1), response(2))
    m = method(tmp_path, llm, budget=2)
    monkeypatch.setattr(m.rng, "choices", lambda *arguments, **keywords: ["Pivot"])
    root = add(m, 1, idea="")
    add(m, 2, parent=root.id, code="def score(x):\n    return 2", idea="child")
    m.run()
    events = read_journal(m.events_path)
    assert events[0]["requested_operator"] == "Pivot" and events[0]["operator"] == "Pivot"
    assert "fallback_reason" not in events[0]["selection"]
    assert events[0]["reference_ids"] == []
    prompt = llm.calls[0][0]
    assert OPERATOR_INSTRUCTIONS["Pivot"] in prompt
    assert "# Reference Nodes" not in prompt and "# Design History" not in prompt


def test_pivot_prompt_shows_reference_cards_and_records_ids(tmp_path, monkeypatch):
    llm = FakeLLM(response(1), response(3))
    m = method(tmp_path, llm, budget=2)
    monkeypatch.setattr(m.rng, "choices", lambda *arguments, **keywords: ["Pivot"])
    root = add(m, 1, idea="root idea")
    add(m, 2, parent=root.id, code="def score(x):\n    return 2", idea="child idea")
    m.run()
    prompt = llm.calls[0][0]
    assert "# Reference Nodes" in prompt and REFERENCE_INTRO in prompt
    assert "Reference 1 | Fitness: 1" in prompt and "Idea: root idea" in prompt
    assert "# Design History" not in prompt and "# Current Algorithm" in prompt
    events = read_journal(m.events_path)
    assert events[0]["operator"] == "Pivot"
    assert events[0]["reference_ids"] == [root.id]
    selection = events[0]["selection"]
    assert set(selection) == {"code_digest", "mean_fitness", "percentile", "attempts",
                              "bonus", "score", "tie_group_size", "formation_candidates"}
    assert selection["percentile"] == 1.0 and selection["mean_fitness"] == 2


def test_pivot_and_fuse_prompts_keep_fitness_direction_after_the_task(tmp_path):
    m = method(tmp_path, budget=1)
    root = add(m, 1, idea="root idea")
    child = add(m, 2, parent=root.id, code="def score(x):\n    return 2", idea="child idea")
    for operator in ("Pivot", "Fuse"):
        text, retained = m.reference_builder.build(child, operator, [root])
        parts = text.split("\n\n\n")
        assert parts[0].startswith("# Task")
        assert parts[1] == "Fitness: higher is better."
        assert parts[2].startswith("# Current Algorithm")
        assert "# Reference Nodes" in parts[3]
        assert OPERATOR_INSTRUCTIONS[operator] in parts[4]
        assert parts[5].startswith("# Output")
        assert [node.id for node in retained] == [root.id]
    refine = m.builder.build(child, "Refine").split("\n\n\n")
    assert refine[0].startswith("# Task") and refine[1] == "Fitness: higher is better."


def test_new_operator_instructions_match_the_design():
    assert OPERATOR_INSTRUCTIONS["Pivot"] == (
        "Identify a limitation of the current approach in relation to the task, "
        "using the reference ideas where helpful. Develop and implement a "
        "competitive alternative based on a different main idea that addresses "
        "this limitation."
    )
    assert OPERATOR_INSTRUCTIONS["Fuse"] == (
        "Compare the current algorithm and the reference ideas to identify "
        "useful ideas that can complement one another. Adapt and combine "
        "selected ideas into a coherent algorithm that aims to improve "
        "performance on the task."
    )


def test_frozen_prompt_parts_match_v1011():
    from llm4ad.method.traceaad_v10_11.errors import OUTPUT as V1011_OUTPUT
    from llm4ad.method.traceaad_v10_11.trajectory import OPERATOR_INSTRUCTIONS as V1011_INSTRUCTIONS
    assert OUTPUT == V1011_OUTPUT
    for operator in ("Init", "Refine", "Tune"):
        assert OPERATOR_INSTRUCTIONS[operator] == V1011_INSTRUCTIONS[operator]
    assert OPERATOR_INSTRUCTIONS["Pivot"] != V1011_INSTRUCTIONS["Pivot"]
    assert OPERATOR_INSTRUCTIONS["Fuse"] != V1011_INSTRUCTIONS["Fuse"]


def test_refine_and_tune_prompts_are_unchanged_from_v1011(tmp_path):
    from llm4ad.method.traceaad_v10_11.trajectory import TrajectoryBuilder as V1011Builder
    m = method(tmp_path, budget=1)
    root = add(m, 1)
    child = add(m, 2, root.id, code="def score(x):\n    return 2")
    legacy = V1011Builder(FakeLLM(), m.task_contract, max_tokens=24576, max_events=8,
                          lookup=m.tree.nodes.get, all_nodes=m.tree.all_nodes)
    for operator in ("Refine", "Tune"):
        assert m.builder.build(child, operator) == legacy.build(child, operator)
    assert m.builder.build_initial() == legacy.build_initial()


def test_mechanism_fingerprint_records_v11_scheduling(tmp_path):
    m = method(tmp_path, budget=1)
    assert m.mechanism["method"] == "v110"
    assert m.mechanism["exploration_c"] == 0.1
    assert m.mechanism["n_references"] == 8
    assert m.mechanism["reference_weighting"] == "reciprocal_rank"
    for removed in ("quality_ess_target", "pivot_uniform_probability", "donor_uniform_probability"):
        assert removed not in m.mechanism


def test_incompatible_checkpoint_is_rejected(tmp_path):
    m = method(tmp_path, FakeLLM(response(1)), budget=1)
    m.run()
    state = json.loads(m.state_path.read_text())
    state["mechanism"] = {"method": "v1011"}
    m.state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="checkpoint configuration differs"):
        method(tmp_path, FakeLLM(response(9)), budget=1).run()


def test_service_routing_update_keeps_the_checkpoint_loadable(tmp_path):
    """prepare_resume rewrites only mechanism.llm routing fields before a resume."""
    m = method(tmp_path, FakeLLM(response(1), response(2)), budget=2)
    m.run()
    state = json.loads(m.state_path.read_text())
    state["mechanism"]["llm"].update({"base_url": "http://127.0.0.1:9999/v1",
                                      "model": "equivalent-service"})
    m.state_path.write_text(json.dumps(state))
    resumed = method(tmp_path, FakeLLM(base_url="http://127.0.0.1:9999/v1",
                                       model="equivalent-service"), budget=2)
    resumed.run()
    assert resumed.budget_used == 2 and resumed.llm.calls == []


def test_runs_function_through_template_and_parses_like_v1011(tmp_path):
    m = method(tmp_path, FakeLLM(response(7)), budget=1)
    assert m.parse_response("```python\ndef score(x):\n    return 1\n```") is None
    parsed = m.parse_response(
        "Idea: constant score\nCode:\n```python\nimport math\n\ndef helper(x):\n    "
        "return math.floor(x) + 1\n\ndef score(x):\n    return helper(x)\n```"
    )
    assert parsed[0] == "constant score" and "def helper" in parsed[2]
    m.run()
    assert m.tree.best().fitness == 7 and "def score" in m.tree.best().code


def test_repair_prompt_includes_evaluator_error_details():
    prompt = repair_prompt(
        "Task contract",
        "Idea: broken\nCode:\n```python\ndef score(x):\n    return x\n```",
        {"error_type": "ValueError", "error": "array dimensions do not match",
         "reason": "runtime_error"},
    )
    assert "ValueError: array dimensions do not match" in prompt
    assert "Keep the Idea to no more than 200 words." in prompt


def test_history_preserves_the_full_bounded_idea(tmp_path):
    m = method(tmp_path, budget=1)
    root = add(m, 1)
    child = add(m, 2, root.id, code="def score(x):\n    return x")
    child.idea = " ".join(["long"] * 200)
    assert m.builder.build(child, "Refine").count("long") == 200


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


def test_transport_failure_resumes_the_same_request_and_replays_counters(tmp_path):
    m = method(tmp_path, FakeLLM(response(1), response("missing"), RuntimeError("offline")), budget=3)
    with pytest.raises(RuntimeError, match="offline"):
        m.run()
    restored = method(tmp_path, FakeLLM(response(3)), budget=3)
    restored.run()
    assert restored.llm.calls[0][0] == m.llm.calls[2][0]
    assert [call["call_id"] for call in read_journal(m.llm_calls_path)] == ["1:1", "2:1", "3:1", "3:2"]
    assert read_journal(m.events_path)[-1]["repair_of"] == 2
    assert restored.parent_selection_counts == {0: 2}
    assert restored.codebook.attempts(code_key(m.tree.nodes[0].code)) == 2


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


def test_persistence_keeps_single_copies_and_replays_code_statistics(tmp_path):
    m = method(tmp_path, FakeLLM(response(1), response(2)), budget=2)
    m.run()
    calls = read_journal(m.llm_calls_path)
    assert calls and all("prompt" not in call for call in calls)
    assert all(call.get("prompt_hash") and call.get("response") for call in calls)
    state = json.loads(m.state_path.read_text())
    assert "nodes" not in state and len(state["code_attempts"]) == len(m.codebook.entries)
    assert sum(state["code_attempts"].values()) == state["step_counter"]
    nodes = read_journal(m.nodes_path)
    assert [node["id"] for node in nodes] == [0, 1]
    assert all(node["donor_id"] is None for node in nodes)

    resumed = method(tmp_path, FakeLLM(), budget=2)
    resumed.run()
    assert len(resumed.tree.nodes) == 2 and resumed.budget_used == 2
    assert resumed.llm.calls == [] and resumed.step_counter == 1
    assert resumed.codebook.attempts(code_key(nodes[0]["code"])) == 1
    assert resumed.parent_selection_counts == {0: 1}


@pytest.mark.parametrize("position", [
    "before_generation", "after_generation", "during_evaluation", "after_evaluation",
    "after_node", "after_event", "after_checkpoint",
])
def test_recovery_settles_each_candidate_exactly_once(tmp_path, monkeypatch, position):
    """Acceptance check for the checkpoint-recovery contract.

    A fake run stops at each interruption position during the second candidate;
    the resumed run must call the evaluator, count budget, create nodes, and
    settle n/T exactly once per candidate.
    """
    evaluation_calls = []
    m = method(tmp_path, FakeLLM(response(1), response(2)), budget=2)
    original_evaluate = m.secure.evaluate_program_with_details

    def probe(program):
        evaluation_calls.append(program)
        if position == "during_evaluation" and len(evaluation_calls) == 2:
            raise KeyboardInterrupt
        return original_evaluate(program)
    monkeypatch.setattr(m.secure, "evaluate_program_with_details", probe)

    if position == "before_generation":
        original = m._generate_pending

        def crash():
            if m.pending["candidate_id"] == 2:
                raise KeyboardInterrupt
            original()
        monkeypatch.setattr(m, "_generate_pending", crash)
    elif position == "after_generation":
        original = m._evaluate_pending

        def crash(parsed):
            if m.pending["candidate_id"] == 2:
                raise KeyboardInterrupt
            return original(parsed)
        monkeypatch.setattr(m, "_evaluate_pending", crash)
    elif position == "after_evaluation":
        original = m._settle_node

        def crash(outcome, parsed):
            if m.pending["candidate_id"] == 2:
                raise KeyboardInterrupt
            return original(outcome, parsed)
        monkeypatch.setattr(m, "_settle_node", crash)
    elif position == "after_node":
        original = m._append_record

        def crash(path, record):
            original(path, record)
            if path == m.nodes_path and record.get("evaluation_id") == 2:
                raise KeyboardInterrupt
        monkeypatch.setattr(m, "_append_record", crash)
    elif position == "after_event":
        original = m._save_state

        def crash():
            if m.pending is not None and m.pending["candidate_id"] == 2:
                raise KeyboardInterrupt  # event journal leads the checkpoint
            original()
        monkeypatch.setattr(m, "_save_state", crash)
    elif position == "after_checkpoint":
        original = m._save_state

        def crash():
            original()
            if m.pending is not None and m.pending["candidate_id"] == 2:
                raise KeyboardInterrupt  # checkpoint saved, pending file not yet cleaned
        monkeypatch.setattr(m, "_save_state", crash)

    with pytest.raises(KeyboardInterrupt):
        m.run()
    if position in ("after_event", "after_checkpoint"):
        persisted = json.loads(m.state_path.read_text())["completed_attempts"]
        assert persisted == (1 if position == "after_event" else 2)

    resumed = method(tmp_path, FakeLLM(response(2)), budget=2)
    resumed_evaluate = resumed.secure.evaluate_program_with_details

    def resumed_probe(program):
        evaluation_calls.append(program)
        return resumed_evaluate(program)
    monkeypatch.setattr(resumed.secure, "evaluate_program_with_details", resumed_probe)

    if position == "during_evaluation":
        with pytest.raises(UnknownEvaluation):
            resumed.run()
        assert resumed.budget_used == 1
        assert len(evaluation_calls) == 2
        return

    resumed.run()
    assert len(evaluation_calls) == 2 and resumed.budget_used == 2
    nodes = read_journal(resumed.nodes_path)
    assert [node["id"] for node in nodes] == [0, 1]
    events = read_journal(resumed.events_path)
    assert [event["candidate_id"] for event in events] == [1, 2]
    assert resumed.step_counter == 1
    assert resumed.codebook.attempts(code_key(nodes[0]["code"])) == 1
    assert json.loads(resumed.summary_path.read_text())["status"] == "finished"


def test_mixed_run_journals_replay_consistently(tmp_path):
    """Duplicates, failures, repairs and invalid outputs interleave; journals replay exactly."""
    llm = FakeLLM(response(1), response(2), response(1), response("missing"),
                  response(3), "bad output", response(4), response(5), response(6))
    m = method(tmp_path, llm, budget=6)
    m.run()
    events = read_journal(m.events_path)
    assert [event["status"] for event in events] == [
        "ok", "ok", "ok", "eval_failed", "ok", "invalid_output", "ok"]
    assert len({event["candidate_id"] for event in events}) == len(events)
    assert m.budget_used == max(event["evaluation_id"] for event in events
                                if event["evaluation_id"] is not None)
    assert m.step_counter == sum(1 for event in events if event["parent_id"] is not None)
    nodes = read_journal(m.nodes_path)
    assert [node["evaluation_id"] for node in nodes] == [1, 2, 3, 5, 6]
    for key, entry in m.codebook.entries.items():
        logged = [node for node in nodes if code_key(node["code"]) == key]
        assert entry["scores"] == [node["fitness"] for node in logged]
    settled = sum(1 for event in events if event["parent_id"] is not None)
    assert sum(m.codebook.attempts(key) for key in m.codebook.keys()) == settled

    fresh = method(tmp_path, FakeLLM(), budget=6)
    fresh.run()
    assert fresh.llm.calls == []
    assert (fresh.budget_used, fresh.step_counter, fresh.completed_attempts) == (
        m.budget_used, m.step_counter, m.completed_attempts)
    assert fresh.codebook.attempts_table() == m.codebook.attempts_table()
    assert fresh.parent_selection_counts == m.parent_selection_counts
    assert fresh.tree.best().fitness == m.tree.best().fitness


def test_read_journal_ignores_torn_tail_but_rejects_midfile_corruption(tmp_path):
    path = tmp_path / "journal.jsonl"
    path.write_bytes(b'{"a": 1}\n{"b": 2}\n{"torn": ')
    assert read_journal(path) == [{"a": 1}, {"b": 2}]
    truncate_torn_tail(path)
    assert path.read_bytes() == b'{"a": 1}\n{"b": 2}\n'
    path.write_bytes(b'{"a": 1}\n{"broken": [}\n{"c": 3}\n')
    with pytest.raises(json.JSONDecodeError):
        read_journal(path)


@pytest.mark.parametrize("target", ["nodes", "events", "llm_calls"])
def test_torn_final_journal_write_is_recovered(tmp_path, monkeypatch, target):
    """A half-written final journal line must not block the pending-based recovery."""
    evaluation_calls = []
    m = method(tmp_path, FakeLLM(response(1), response(2)), budget=2)
    original_evaluate = m.secure.evaluate_program_with_details

    def probe(program):
        evaluation_calls.append(program)
        return original_evaluate(program)
    monkeypatch.setattr(m.secure, "evaluate_program_with_details", probe)
    original_append = m._append_record

    def torn(path, record):
        torn_now = (
            (target == "nodes" and path == m.nodes_path and record.get("evaluation_id") == 2) or
            (target == "events" and path == m.events_path and record.get("candidate_id") == 2) or
            (target == "llm_calls" and path == m.llm_calls_path and record.get("candidate_id") == 2))
        if torn_now:
            line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line[:len(line) // 2])  # torn write: no trailing newline
            raise KeyboardInterrupt
        original_append(path, record)
    monkeypatch.setattr(m, "_append_record", torn)
    with pytest.raises(KeyboardInterrupt):
        m.run()

    resumed = method(tmp_path, FakeLLM(response(2)), budget=2)
    resumed_evaluate = resumed.secure.evaluate_program_with_details

    def resumed_probe(program):
        evaluation_calls.append(program)
        return resumed_evaluate(program)
    monkeypatch.setattr(resumed.secure, "evaluate_program_with_details", resumed_probe)
    resumed.run()
    assert len(evaluation_calls) == 2 and resumed.budget_used == 2
    assert [node["id"] for node in read_journal(resumed.nodes_path)] == [0, 1]
    assert [event["candidate_id"] for event in read_journal(resumed.events_path)] == [1, 2]
    assert resumed.step_counter == 1
    assert resumed.codebook.attempts(code_key(read_journal(resumed.nodes_path)[0]["code"])) == 1
    assert json.loads(resumed.summary_path.read_text())["status"] == "finished"


def test_fuse_over_capacity_falls_back_when_refine_prompt_fits(tmp_path, monkeypatch):
    """Empty-reference Fuse may exceed capacity while the Refine fallback fits."""
    llm = FakeLLM(response(2))
    m = method(tmp_path, llm, budget=1)
    monkeypatch.setattr(m.rng, "choices", lambda *arguments, **keywords: ["Fuse"])
    root = add(m, 1, idea="")  # no formation history and an empty reference pool
    capacity = m.builder.count(m.builder.build(root, "Refine"))
    minimal_fuse, _ = m.reference_builder.build(root, "Fuse", [])
    assert m.reference_builder.count(minimal_fuse) > capacity  # the reported failure mode
    m.max_input_tokens = capacity
    m.builder.max_tokens = capacity
    m.reference_builder.max_tokens = capacity
    m.run()
    events = read_journal(m.events_path)
    assert events[0]["operator"] == "Refine"
    assert events[0]["selection"]["fallback_reason"] == "reference_pool_empty"
    assert events[0]["reference_ids"] == []
    assert OPERATOR_INSTRUCTIONS["Refine"] in llm.calls[0][0]


def test_stale_pending_cleanup_saves_the_checkpoint_before_deleting(tmp_path, monkeypatch):
    """Interrupting the cleanup between its two steps must not strand the old RNG."""
    m = method(tmp_path, FakeLLM(response(1), response(2)), budget=2)
    original = m._save_state

    def crash_after_event():
        if m.pending is not None and m.pending["candidate_id"] == 2:
            raise KeyboardInterrupt
        original()
    monkeypatch.setattr(m, "_save_state", crash_after_event)
    with pytest.raises(KeyboardInterrupt):
        m.run()

    first = method(tmp_path, FakeLLM(), budget=2)
    original_first_save = first._save_state

    def stop_after_save():
        original_first_save()
        raise KeyboardInterrupt  # inside the cleanup: checkpoint saved, pending kept
    monkeypatch.setattr(first, "_save_state", stop_after_save)
    with pytest.raises(KeyboardInterrupt):
        first.run()
    assert first.pending_path.exists()
    state = json.loads(first.state_path.read_text())
    pending = json.loads(first.pending_path.read_text())
    assert state["completed_attempts"] == 2
    assert state["rng_state"] == pending["rng_state"]

    second = method(tmp_path, FakeLLM(response(3)), budget=2)
    second.run()
    assert second.llm.calls == [] and second.budget_used == 2
    assert json.loads(second.summary_path.read_text())["status"] == "finished"
