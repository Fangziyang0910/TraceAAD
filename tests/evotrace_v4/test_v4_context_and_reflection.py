from copy import deepcopy
import json

import pytest

from evotrace_v4 import Task
from evotrace_v4.context import EXPLORE_CONTEXT_CHARS, build_evolution_context, evolution_prompt
from evotrace_v4.models import CANDIDATE_RESPONSE_FORMAT, Candidate, Edge, FormatError, Node, ROOT_ID, TreePopulation
from evotrace_v4.reflection import build_reflection_prompt, reflect_and_apply
from evotrace_v4.topology import PathContraction, SubtreePruning

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


def test_candidate_schema_and_parser_use_analysis_code_and_design_in_order():
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
    with pytest.raises(FormatError):
        Candidate.parse('{"analysis": "Plan", "design": "Reversed order", "code": "def solve(x): return x"}')


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

    # When S \\ {p} is empty (explore_ids=()), only the current algorithm card remains.
    solo_ctx, _ = build_evolution_context(tree, parent, "Explore", explore_ids=())
    assert "[Other Existing Approaches]" not in solo_ctx and solo_ctx.count("```python") == 1

    # include_experiences=False mutes edge and node experiences cleanly.
    muted, _ = build_evolution_context(tree, parent, "Explore", explore_ids=(rep_best,), include_experiences=False)
    assert not any(note in muted for note in notes)
    assert f"Compute offset {rep_best}" in muted and f"Compute offset {parent}" in muted

    with pytest.raises(ValueError, match="parent_id"):
        build_evolution_context(tree, ROOT_ID, "Explore")


def test_explore_skips_reference_cards_that_would_exceed_the_input_limit():
    """端点输入上限是硬约束：放不下的代表卡片按抽样顺序略去，主父代卡片始终保留。"""
    tree = TreePopulation()
    parent = add(tree, 0, 6.0)
    body = "".join(f"    v{k} = x + {k}\n" for k in range(1800))
    reps = [add(tree, 0, 5.0 + k, code=f"def solve(x):\n{body}    return x + {k}\n") for k in range(3)]
    context, _ = build_evolution_context(tree, parent, "Explore", explore_ids=tuple(reps))
    assert len(context) <= EXPLORE_CONTEXT_CHARS
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
    assert "Experience from earlier attempts: CURRENT_ATTEMPTS_NOTE" in ctx_ref
    assert f"Compute offset {tried}" in ctx_ref
    assert ctx_ref.index("[Development of the Current Algorithm]") < ctx_ref.index("[Attempts from the Current Algorithm]")
    assert ctx_ref.index("[Attempts from the Current Algorithm]") < ctx_ref.index("[Current Algorithm]")
    assert ctx_ref.count("```python") == 1 and ctx_ref.endswith("```")
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


def test_reflection_distinguishes_explore_origin_and_preserves_experience_on_failure():
    tree = TreePopulation()
    a = add(tree, 0, 10.0, "def f(): return 0")
    b = add(tree, a, 12.0, "def f(): return 1", op="Explore")
    c = add(tree, b, 8.0, "def f(): return 2", op="Refine")
    tree.nodes[b].experience = "Tried jumping from a."

    region = PathContraction(a, c, (b,))
    prompt, _ = build_reflection_prompt(tree, region, PROBLEM)
    assert f"Algorithm {b}, generated from Algorithm {a} aiming at a different core computation: 10 -> 12 (regressed)" in prompt

    class Broken:
        def complete(self, *args, **kwargs):
            raise TimeoutError("down")

    after, calls = reflect_and_apply(tree, (region,), model=Broken(), problem=PROBLEM)
    assert calls[0]["status"] == "error"
    assert after.edges[c].contracted and after.edges[c].parent_id == a
    assert after.edges[c].experience == "Tried jumping from a."


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
    with pytest.raises(ValueError, match="virtual root"):
        build_reflection_prompt(tree, region, PROBLEM)


def test_refine_and_explore_instructions_allow_procedure_change():
    tree = TreePopulation()
    init_a = add(tree, 0, 9.0)
    parent = add(tree, init_a, 6.0)
    _, refine = build_evolution_context(tree, parent, "Refine")
    assert "preserving its core" not in refine
    assert "or its procedure as well as its parameters" in refine
    _, explore = build_evolution_context(tree, parent, "Explore")
    assert "not a structure you must preserve" in explore
    assert "not another weighting of the same local features" in explore
    assert "construct and improve a tentative solution" in explore
