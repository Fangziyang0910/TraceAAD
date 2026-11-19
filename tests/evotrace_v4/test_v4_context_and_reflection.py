from copy import deepcopy
from dataclasses import asdict
import json

import pytest

from evotrace_v4 import Task
from evotrace_v4.context import CONTEXT_CHARS, build_evolution_context, evolution_prompt, init_prompt, repair_program_prompt
from evotrace_v4.models import (
    CANDIDATE_RESPONSE_FORMAT, Candidate, Edge, Evaluation, FormatError, Node, Program, ROOT_ID, TreePopulation,
)
from evotrace_v4.reflection import EXPERIENCE_SCHEMA, build_reflection_prompt, reflect_and_apply
from evotrace_v4.topology import PathContraction, SubtreePruning, TopologyConfig, bare_code, plan_simplification

PROBLEM = "[Problem]\nMinimize.\nScore: lower is better."


def add(tree, parent, fit, code=None, op="Refine"):
    i = max(tree.nodes, default=0) + 1
    src = code if code is not None else f"def solve(x):\n    return x + {i}\n"
    tree.add(
        Node(i, src, f"Compute offset {i}", fit, failure="ZeroDivisionError" if fit is None else None),
        Edge(parent, i, None if parent == ROOT_ID else op),
        evaluations=1,
    )
    return i


def test_candidate_schema_orders_analysis_code_and_design_and_the_parser_reads_the_three_fields():
    schema = CANDIDATE_RESPONSE_FORMAT["json_schema"]["schema"]
    assert schema["required"] == ["analysis", "code", "design"]
    assert list(schema["properties"]) == ["analysis", "code", "design"]
    assert schema["additionalProperties"] is False

    cand = Candidate.parse(json.dumps({
        "analysis": "Plan a greedy step", "code": "def solve(x):\n    return x\n", "design": "  Greedy   step ",
    }))
    assert cand == Candidate(analysis="Plan a greedy step", code="def solve(x):\n    return x\n", design="Greedy step")
    with pytest.raises(FormatError):
        Candidate.parse(json.dumps({"code": "def solve(x): return x", "design": "Greedy step"}))
    with pytest.raises(FormatError):
        Candidate.parse(json.dumps({"analysis": "", "code": "def solve(x): return x", "design": "Greedy step"}))
    reordered = Candidate.parse('{"analysis": "Plan", "design": "Reversed order", "code": "def solve(x): return x"}')
    assert reordered.design == "Reversed order"


def test_displayed_code_keeps_the_original_writing_without_docstrings_or_comments():
    """ast.unparse 把 min(x for x in y) 改写成 min((x for x in y))，模型照抄时会漏写右括号。"""
    code = 'def solve(x):\n    """Doc."""\n    # pick the best\n    return float(min(v + 1 for v in x))  # tail\n'
    assert bare_code(code) == "def solve(x):\n    return float(min(v + 1 for v in x))"
    assert bare_code('def f():\n    """Only a docstring."""\n') == "def f():\n    pass"


def test_repair_shows_the_executed_code_and_the_failing_line_with_its_numbers():
    """评测器用 exec 运行源码时，报错只有 "<string>" 行号；修复提示要能对上这一行。"""
    code = 'import numpy as np\n\ndef solve(x):\n    """Doc."""\n    # widen\n    return np.zeros(3) + np.ones((1, 4))\n'
    failure = (
        "Traceback (most recent call last):\n"
        '  File "/srv/benchmarks/shared/fixed_evaluation.py", line 42, in evaluate_program\n'
        "    _, objective = self.SOLVER(data, namespace[self.FUNCTION_NAME])\n"
        '  File "<string>", line 6, in solve\n'
        "ValueError: operands could not be broadcast together with shapes (3,) (1,4)\n"
    )
    program = Program(1, code, "Add arrays", "Init", 0, None, Evaluation("error", failure=failure))
    prompt = repair_program_prompt(Task("Minimize", "def solve(x): pass", lambda s, f: f(1), 2), program)
    assert code.rstrip() in prompt
    assert ('File "<string>", line 6, in solve\n    return np.zeros(3) + np.ones((1, 4))\n'
            "ValueError: operands could not be broadcast together with shapes (3,) (1,4)") in prompt
    assert "fixed_evaluation" not in prompt


def test_explore_shows_parent_and_diverse_cards_with_their_own_experience_and_code():
    tree = TreePopulation()
    init_a = add(tree, 0, 9.0)
    parent = add(tree, init_a, 6.0, op="Explore")
    tried_child = add(tree, parent, 7.5, op="Refine")
    rep_best = add(tree, 0, 5.0)
    rep_far = add(tree, 0, 8.0)

    tree.nodes[init_a].experience = "ANCESTOR_NOTE"
    tree.edges[parent].experience = "PARENT_STAGE_NOTE"
    tree.nodes[parent].experience = "PARENT_ATTEMPTS_NOTE"
    tree.edges[rep_best].experience = "REP_STAGE_NOTE"
    tree.nodes[rep_best].experience = "REP_ATTEMPTS_NOTE"
    notes = ("PARENT_STAGE_NOTE", "PARENT_ATTEMPTS_NOTE", "REP_STAGE_NOTE", "REP_ATTEMPTS_NOTE")

    task = Task("Minimize", "def solve(x): pass", lambda s, f: f(1), 2)
    prompt = evolution_prompt(task, tree, parent, "Explore", explore_ids=(rep_far, rep_best))
    assert prompt.index('- "analysis":') < prompt.index('- "code":') < prompt.index('- "design":')

    context, _ = build_evolution_context(tree, parent, "Explore", explore_ids=(rep_far, rep_best))
    # The current algorithm comes first; the other approaches follow from best to worst fitness.
    assert context.index("[Current Algorithm]") < context.index("[Other Existing Approaches]")
    assert context.index(f"Compute offset {rep_best}") < context.index(f"Compute offset {rep_far}")
    # Each card shows design, its own in-edge and node experience, and code last.
    assert context.count("```python") == 3
    assert all(context.count(note) == 1 for note in notes)
    parent_card = context[:context.index("[Other Existing Approaches]")]
    assert parent_card.index(f"Compute offset {parent}") < parent_card.index("PARENT_STAGE_NOTE")
    assert parent_card.index("PARENT_ATTEMPTS_NOTE") < parent_card.index("```python")
    # No development path or attempt list: ancestors and attempts of the parent are absent.
    assert "[Development of the Current Algorithm]" not in context
    assert "[Attempts from the Current Algorithm]" not in context
    assert "ANCESTOR_NOTE" not in context
    assert f"Compute offset {init_a}" not in context and f"Compute offset {tried_child}" not in context

    # With no eligible references, only the current algorithm card remains.
    solo_ctx, _ = build_evolution_context(tree, parent, "Explore", explore_ids=())
    assert "[Other Existing Approaches]" not in solo_ctx and solo_ctx.count("```python") == 1

    # include_experiences=False mutes edge and node experiences cleanly.
    muted, _ = build_evolution_context(tree, parent, "Explore", explore_ids=(rep_best,), include_experiences=False)
    assert not any(note in muted for note in notes)
    assert f"Compute offset {rep_best}" in muted and f"Compute offset {parent}" in muted


def test_explore_skips_reference_cards_that_would_exceed_the_input_limit():
    """可选材料按完整树上下文的字符数计入，主父代卡片始终保留。"""
    tree = TreePopulation()
    parent = add(tree, 0, 6.0)
    body = "".join(f"    v{k} = x + {k}\n" for k in range(1800))
    reps = [add(tree, 0, 5.0 + k, code=f"def solve(x):\n{body}    return x + {k}\n") for k in range(3)]
    context, _ = build_evolution_context(tree, parent, "Explore", explore_ids=tuple(reps))
    assert len(context) <= CONTEXT_CHARS
    assert f"Compute offset {parent}" in context and context.count("```python") == 3
    assert f"Compute offset {reps[2]}" not in context


def test_refine_and_crossover_read_the_development_path_and_the_reference_card():
    tree = TreePopulation()
    a = add(tree, 0, 10.0)
    b = add(tree, a, 8.0, op="Explore")
    current = add(tree, b, 6.0, op="Refine")
    tried = add(tree, current, 6.5, op="Refine")
    side = add(tree, a, 7.0, op="Refine")
    reference = add(tree, side, 5.5, op="Refine")
    tree.nodes[a].experience = "A_NOTE"
    tree.edges[b].experience = "EXPLORE_EDGE_NOTE"
    tree.nodes[current].experience = "CURRENT_ATTEMPTS_NOTE"
    tree.nodes[side].experience = "SIDE_NOTE"
    tree.edges[reference].experience = "REF_EDGE_NOTE"
    tree.nodes[reference].experience = "REF_NOTE"
    before = deepcopy(tree)

    ctx_ref, _ = build_evolution_context(tree, current, "Refine")
    assert "Step 1 (aimed at a different core computation): 10 -> 8 (improved)" in ctx_ref
    assert "Step 2, the current algorithm (aimed at improvement): 8 -> 6 (improved)" in ctx_ref
    assert "EXPLORE_EDGE_NOTE" in ctx_ref and "A_NOTE" in ctx_ref
    assert "Local reflection on earlier attempts: CURRENT_ATTEMPTS_NOTE" in ctx_ref
    assert f"Compute offset {tried}" in ctx_ref
    assert ctx_ref.index("[Development of the Current Algorithm]") < ctx_ref.index("[Attempts from the Current Algorithm]")
    assert ctx_ref.index("[Attempts from the Current Algorithm]") < ctx_ref.index("[Current Algorithm]")
    assert ctx_ref.count("```python") == 2 and ctx_ref.endswith("```")
    previous = ctx_ref.split("[Previous Retained Implementation]")[1].split("[Current Algorithm]")[0]
    assert "return x + 2" in previous and "fitness = 8" in previous
    assert "SIDE_NOTE" not in ctx_ref and "REF_NOTE" not in ctx_ref

    ctx_cross, _ = build_evolution_context(tree, current, "Crossover", reference)
    other = ctx_cross[ctx_cross.index("[Other Algorithm]"):ctx_cross.index("[Current Algorithm]")]
    assert f"Compute offset {reference}" in other and "REF_EDGE_NOTE" in other and "REF_NOTE" in other
    assert "SIDE_NOTE" not in ctx_cross  # Only the reference's own experience, not its development path.
    assert ctx_cross.count("A_NOTE") == 1
    assert ctx_cross.count("```python") == 2 and ctx_cross.endswith("```")

    # A reference on the development path or among the attempts is shown once, as the reference card.
    ctx_anc, _ = build_evolution_context(tree, current, "Crossover", a)
    assert ctx_anc.count("A_NOTE") == 1
    ctx_child, _ = build_evolution_context(tree, current, "Crossover", tried)
    assert ctx_child.count(f"Compute offset {tried}") == 1
    assert tree == before


def test_refine_and_crossover_read_each_original_edge_intent_in_lineage_order():
    tree = TreePopulation()
    first = add(tree, ROOT_ID, 10.0)
    mid = add(tree, first, 9.0)
    parent = add(tree, mid, 8.0)
    attempt = add(tree, parent, 9.0)
    donor = add(tree, ROOT_ID, 7.0)
    for i in (first, mid, parent, attempt, donor):
        tree.nodes[i].analysis = f"INTENT_{i}."

    for op, ref in [("Refine", None), ("Crossover", donor), ("Crossover", attempt), ("Crossover", first)]:
        context, _ = build_evolution_context(tree, parent, op, ref)
        assert context.count("INTENT_") == 3
        positions = []
        for i in (first, mid, parent):
            positions.extend((context.index(tree.nodes[i].analysis), context.index(f"Reported design: Compute offset {i}")))
        assert positions == sorted(positions)
    explore, _ = build_evolution_context(tree, parent, "Explore", explore_ids=(first, attempt, donor))
    assert "INTENT_" not in explore

    # Init also has an original incoming edge. Its design intent predates reflection.
    initial, _ = build_evolution_context(tree, first, "Refine", include_experiences=False)
    assert "designed from scratch" in initial and tree.nodes[first].analysis in initial
    saved = asdict(tree)
    del saved["nodes"][first]["analysis"]
    legacy, _ = build_evolution_context(TreePopulation.from_dict(saved), first, "Refine")
    assert "Reported intent" not in legacy


def test_path_reflection_labels_steps_without_ranking_them_against_the_end_and_keeps_experience_on_failure():
    tree = TreePopulation()
    a = add(tree, 0, 10.0, "def f(): return 0")
    b = add(tree, a, 12.0, "def f(): return 1", op="Explore")
    c = add(tree, b, 8.0, "def f(): return 2", op="Refine")
    tree.nodes[b].experience = "Tried jumping from a."

    region = PathContraction(a, c, (b,))
    prompt, _ = build_reflection_prompt(tree, region, PROBLEM)
    assert f"Algorithm {b}, on the path, generated from Algorithm {a} aiming at a different core computation: 10 -> 12 (regressed)" in prompt
    assert f"Algorithm {c}, the end of the path" in prompt
    assert " than Algorithm" not in prompt  # Stepping stones are not marked worse than the algorithm they built.

    class Broken:
        def complete(self, *args, **kwargs):
            raise TimeoutError("down")

    after, calls = reflect_and_apply(tree, (region,), model=Broken(), problem=PROBLEM)
    assert calls[0]["status"] == "error"
    assert after.edges[c].contracted and after.edges[c].parent_id == a
    assert after.edges[c].experience == "Tried jumping from a."


def test_subtree_reflection_compares_attempts_from_removed_parents_with_the_starting_algorithm():
    tree = TreePopulation()
    p = add(tree, 0, 7.0)
    worse = add(tree, p, 8.0, op="Explore")
    recovered = add(tree, worse, 7.5)  # Better than its removed parent, still worse than p.
    prompt, _ = build_reflection_prompt(tree, SubtreePruning(p, (worse, recovered)), PROBLEM)
    assert f"Algorithm {worse}, generated from Algorithm {p} aiming at a different core computation: 7 -> 8 (regressed)" in prompt
    assert f"attempts to improve Algorithm {p}" not in prompt
    assert f"Algorithm {recovered}, generated from Algorithm {worse} aiming to improve it: 8 -> 7.5 (improved); worse than Algorithm {p}" in prompt


@pytest.mark.parametrize("kind", ["node", "edge"])
@pytest.mark.parametrize(("prior", "reply", "expected", "status"), [
    ("", "", "", "success"),
    ("Try a sweep.", "", "Try a sweep.", "success"),
    ("Try a sweep.", "The actual sweep beat the starting design.", "The actual sweep beat the starting design.", "success"),
    ("Try a sweep.", None, "Try a sweep.", "error"),
])
def test_reflection_updates_the_experience_seen_by_the_next_parent(kind, prior, reply, expected, status):
    tree = TreePopulation()
    start = add(tree, ROOT_ID, 8.0)
    mid = add(tree, start, 9.0)
    tree.nodes[mid].experience = prior
    if kind == "edge":
        target = add(tree, mid, 7.0)
        tree.edges[target].experience = prior
        region = PathContraction(start, target, (mid,))
    else:
        target = start
        tree.nodes[target].experience = prior
        region = SubtreePruning(start, (mid,))

    tree.nodes[target].analysis = "Target's original design intent."
    tree.nodes[mid].analysis = "Intermediate attempt's intent."

    class Model:
        def complete(self, prompt, **kwargs):
            assert "about 120 words" not in prompt
            assert tree.nodes[mid].analysis in prompt
            if kind == "edge":
                assert tree.nodes[target].analysis in prompt
            return json.dumps({"analysis": "ANALYSIS_ONLY", "experience": reply})

    after, (record,) = reflect_and_apply(tree, (region,), model=Model(), problem=PROBLEM)
    assert record["status"] == status
    assert record["experience"] == expected
    context, _ = build_evolution_context(after, target, "Refine")
    if expected:
        assert context.count(expected) == 1
    else:
        assert "Local reflection on" not in context
    if prior and prior != expected:
        assert prior not in context
    assert "ANALYSIS_ONLY" not in context
    intent = tree.nodes[target].analysis
    if kind == "node":
        assert context.count(intent) == 1  # Pruning outgoing branches preserves the original incoming edge.
    else:
        assert intent not in context
        cross, _ = build_evolution_context(after, target, "Crossover", start)
        assert intent not in cross
        if expected:
            assert cross.count(expected) == 1
        muted, _ = build_evolution_context(after, target, "Refine", include_experiences=False)
        assert intent not in muted and "Local reflection on" not in muted
    assert "direction" not in record
    restored = TreePopulation.from_dict(asdict(after))
    leaf = add(restored, target, 6.0)
    restored.nodes[leaf].analysis = "New step's intent."
    if kind == "edge":
        # The contracted incoming edge is now upstream of a new original edge.
        for op, ref in [("Refine", None), ("Crossover", target)]:
            context, _ = build_evolution_context(restored, leaf, op, ref)
            assert intent not in context and context.count(restored.nodes[leaf].analysis) == 1
            if expected:
                assert context.count(expected) == 1
                assert context.index(expected) < context.index(restored.nodes[leaf].analysis)
    next_region = (PathContraction(start, leaf, (target,)) if kind == "edge"
                   else SubtreePruning(target, (leaf,)))
    next_prompt, earlier = build_reflection_prompt(restored, next_region, PROBLEM)
    assert expected in next_prompt
    assert earlier == ([expected] if expected else [])
    assert restored.nodes[leaf].analysis in next_prompt
    assert intent not in next_prompt  # A contracted step or the anchor for branch pruning.



def test_reflection_schema_and_saved_tree_round_trip():
    assert EXPERIENCE_SCHEMA["json_schema"]["schema"]["required"] == ["analysis", "experience"]
    tree = TreePopulation()
    node_id = add(tree, ROOT_ID, 7.0)
    tree.nodes[node_id].experience = "Original note"
    restored = TreePopulation.from_dict(json.loads(json.dumps(asdict(tree))))
    assert restored == tree
    prompt, _ = build_evolution_context(restored, node_id, "Refine")
    assert "Original note" in prompt


def test_same_round_contraction_and_pruning_keep_the_shared_endpoints_notes_separate():
    tree = TreePopulation()
    start = add(tree, ROOT_ID, 9.0)
    mid = add(tree, start, 8.0)
    end = add(tree, mid, 7.0)
    add(tree, end, 8.5)
    tree.nodes[mid].experience = "Earlier side attempt."
    tree.edges[end].experience = "Earlier path step."
    tree.nodes[end].experience = "Earlier endpoint attempt."
    regions = plan_simplification(tree, TopologyConfig(region_size=1), reps=(start, end))

    class Model:
        def complete(self, prompt, **kwargs):
            if "Earlier path step." in prompt:
                assert "Earlier side attempt." in prompt
                assert "Earlier endpoint attempt." not in prompt
                return '{"analysis": "interrupted'
            assert "Earlier endpoint attempt." in prompt and "Earlier side attempt." not in prompt
            return json.dumps({"analysis": "Local update.", "experience": "Revised endpoint note."})

    updated, calls = reflect_and_apply(tree, regions, model=Model(), problem=PROBLEM)
    assert {(c["kind"], c["status"]) for c in calls} == {("edge", "error"), ("node", "success")}
    assert updated.edges[end].experience == "Earlier side attempt. Earlier path step."
    assert updated.nodes[end].experience == "Revised endpoint note."
    context, _ = build_evolution_context(updated, end, "Refine")
    assert all(context.count(note) == 1 for note in (updated.edges[end].experience, updated.nodes[end].experience))
    previous = context.split("[Previous Retained Implementation]")[1].split("[Current Algorithm]")[0]
    assert "return x + 1" in previous


def test_subtree_regions_under_the_virtual_root_are_removed_without_reflection():
    tree = TreePopulation()
    kept = add(tree, 0, 5.0)
    weak = [add(tree, 0, 9.0 + k) for k in range(3)]
    failed = add(tree, 0, None)
    region = SubtreePruning(ROOT_ID, (*weak, failed))

    class Counting:
        calls = 0

        def complete(self, *args, **kwargs):
            Counting.calls += 1
            return json.dumps({"experience": "unused"})

    after, calls = reflect_and_apply(tree, (region,), model=Counting(), problem=PROBLEM)
    assert Counting.calls == 0 and calls == []
    assert set(after.nodes) == {kept} and not after.nodes[kept].experience


def test_refine_keeps_the_current_implementation_when_the_previous_one_does_not_fit(monkeypatch):
    tree = TreePopulation()
    first = add(tree, 0, 9.0)
    parent = add(tree, first, 6.0)
    monkeypatch.setattr("evotrace_v4.context.CONTEXT_CHARS", 1)
    context, _ = build_evolution_context(tree, parent, "Refine")
    assert "[Previous Retained Implementation]" not in context
    assert "[Current Algorithm]" in context and "return x + 2" in context


def test_optional_reference_fits_only_when_the_whole_rendered_context_fits(monkeypatch):
    tree = TreePopulation()
    parent = add(tree, ROOT_ID, 9.0)
    reference = add(tree, ROOT_ID, 8.0)
    full, _ = build_evolution_context(tree, parent, "Explore", explore_ids=(reference,))
    for limit, expected in [(len(full), full), (len(full) - 1, full.split("\n\n[Other Existing Approaches]")[0])]:
        monkeypatch.setattr("evotrace_v4.context.CONTEXT_CHARS", limit)
        context, _ = build_evolution_context(tree, parent, "Explore", explore_ids=(reference,))
        assert context == expected


def test_identical_displayed_predecessor_code_shares_one_listing_and_keeps_both_scores():
    tree = TreePopulation()
    first = add(tree, ROOT_ID, 1.0000002)
    parent = add(tree, first, 1.0000001, code=tree.nodes[first].code)
    context, _ = build_evolution_context(tree, parent, "Refine")
    assert "1.0000002 -> 1.0000001 (improved)" in context
    assert "fitness = 1.0000002\nSame displayed code" in context
    assert "fitness = 1.0000001\n```python" in context
    assert context.count("```python") == 1
    prompt, _ = build_reflection_prompt(tree, SubtreePruning(first, (parent,)), PROBLEM)
    assert "1.0000002 -> 1.0000001 (improved)" in prompt
    assert "Reported design:" in prompt


def test_init_can_start_without_example():
    task = Task("Minimize", "def solve(x): pass", lambda s, f: f(1), 2)
    assert "example" not in init_prompt(task, {}).lower()
