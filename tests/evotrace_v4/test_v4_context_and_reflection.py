from copy import deepcopy
import json

import pytest

from evotrace_v4 import Task
from evotrace_v4.context import build_evolution_context, evolution_prompt
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


def test_candidate_schema_and_parser_use_code_and_design_without_analysis():
    schema = CANDIDATE_RESPONSE_FORMAT["json_schema"]["schema"]
    assert schema["required"] == ["code", "design"]
    assert list(schema["properties"]) == ["code", "design"]
    assert schema["additionalProperties"] is False

    cand = Candidate.parse(json.dumps({"code": "def solve(x):\n    return x\n", "design": "  Greedy   step "}))
    assert cand == Candidate(code="def solve(x):\n    return x\n", design="Greedy step")
    with pytest.raises(FormatError):
        Candidate.parse(json.dumps({"code": "def solve(x): return x"}))
    with pytest.raises(FormatError):
        Candidate.parse(json.dumps({"analysis": "draft", "code": "def solve(x): return x", "design": "Greedy step"}))
    with pytest.raises(FormatError):
        Candidate.parse('{"design": "Reversed order", "code": "def solve(x): return x"}')


def test_explore_reads_parent_root_experience_and_optional_overview_without_other_rep_code():
    tree = TreePopulation()
    init_a = add(tree, 0, 9.0)
    parent = add(tree, init_a, 6.0, op="Explore")
    tried_child = add(tree, parent, 7.5, op="Refine")
    rep_best = add(tree, 0, 5.0)
    rep_far = add(tree, 0, 8.0)

    tree.root_experience = "ROOT_OBSERVATION"
    tree.edges[parent].experience = "PARENT_STAGE_NOTE"
    tree.nodes[parent].experience = "PARENT_ATTEMPTS_NOTE"
    tree.nodes[rep_best].experience = "OTHER_REP_NODE_SECRET"
    tree.edges[rep_best].experience = "OTHER_REP_EDGE_SECRET"

    task = Task("Minimize", "def solve(x): pass", lambda s, f: f(1), 2)
    prompt = evolution_prompt(task, tree, parent, "Explore", explore_ids=(rep_far, rep_best))
    assert '"analysis"' not in prompt
    assert prompt.index('- "code":') < prompt.index('- "design":')
    assert prompt.index("[Other Initial Attempts]") < prompt.index("[Other Existing Approaches]")
    assert prompt.index("[Other Existing Approaches]") < prompt.index("[Current Algorithm]")
    assert prompt.index("[Current Algorithm]") < prompt.index("[Attempts from the Current Algorithm]")
    assert prompt.index(f"Compute offset {rep_best}") < prompt.index(f"Compute offset {rep_far}")

    assert "ROOT_OBSERVATION" in prompt
    assert "PARENT_STAGE_NOTE" in prompt and "PARENT_ATTEMPTS_NOTE" in prompt
    assert f"Compute offset {tried_child}" in prompt
    assert f"return x + {parent}" in prompt  # Parent's code IS read!
    assert f"return x + {rep_best}" not in prompt and f"return x + {rep_far}" not in prompt
    assert "OTHER_REP_NODE_SECRET" not in prompt and "OTHER_REP_EDGE_SECRET" not in prompt

    # A direct child of parent that is also in explore_ids appears once (under [Attempts]), never duplicated.
    dedup_ctx, _ = build_evolution_context(tree, parent, "Explore", explore_ids=(tried_child, rep_best))
    assert dedup_ctx.count(f"Compute offset {tried_child}") == 1

    # When S \ {p} is empty (explore_ids=()), omit [Other Existing Approaches] cleanly.
    solo_ctx, _ = build_evolution_context(tree, parent, "Explore", explore_ids=())
    assert "[Other Existing Approaches]" not in solo_ctx
    assert "ROOT_OBSERVATION" in solo_ctx and f"return x + {parent}" in solo_ctx

    # include_experiences=False mutes root, edge, and node experiences cleanly.
    muted, _ = build_evolution_context(tree, parent, "Explore", explore_ids=(rep_best,), include_experiences=False)
    for note in ("ROOT_OBSERVATION", "PARENT_STAGE_NOTE", "PARENT_ATTEMPTS_NOTE"):
        assert note not in muted
    assert f"Compute offset {rep_best}" in muted and f"return x + {parent}" in muted

    with pytest.raises(ValueError, match="parent_id"):
        build_evolution_context(tree, ROOT_ID, "Explore")


def test_refine_and_crossover_record_explore_steps_and_contracted_edges_along_lineage():
    tree = TreePopulation()
    a = add(tree, 0, 10.0)
    b = add(tree, a, 8.0, op="Explore")
    current = add(tree, b, 6.0, op="Refine")
    reference = add(tree, a, 5.5, op="Refine")
    tree.root_experience = "ROOT_BRANCH_NOTE"
    tree.nodes[a].experience = "A_NOTE"
    tree.edges[b].experience = "EXPLORE_EDGE_NOTE"
    tree.nodes[reference].experience = "REF_NOTE"
    before = deepcopy(tree)

    ctx_ref, _ = build_evolution_context(tree, current, "Refine")
    assert "Experience from other initial attempts: ROOT_BRANCH_NOTE" in ctx_ref
    assert ctx_ref.index("ROOT_BRANCH_NOTE") < ctx_ref.index("First version")
    assert "Step 1 (rethought with a different core computation): 10 -> 8 (improved)" in ctx_ref
    assert "EXPLORE_EDGE_NOTE" in ctx_ref and "A_NOTE" in ctx_ref

    muted_ref, _ = build_evolution_context(tree, current, "Refine", include_experiences=False)
    assert "ROOT_BRANCH_NOTE" not in muted_ref

    ctx_cross, _ = build_evolution_context(tree, current, "Crossover", reference)
    assert ctx_cross.count("ROOT_BRANCH_NOTE") == 1
    assert ctx_cross.count("A_NOTE") == 1  # Shared ancestor experience shown once!
    assert "REF_NOTE" in ctx_cross
    assert ctx_cross.count("```python") == 2
    assert tree == before


def test_reflection_distinguishes_explore_origin_and_preserves_experience_on_failure():
    tree = TreePopulation()
    a = add(tree, 0, 10.0, "def f(): return 0")
    b = add(tree, a, 12.0, "def f(): return 1", op="Explore")
    c = add(tree, b, 8.0, "def f(): return 2", op="Refine")
    tree.nodes[b].experience = "Tried jumping from a."

    region = PathContraction(a, c, (b,))
    prompt, _ = build_reflection_prompt(tree, region, PROBLEM)
    assert f"Algorithm {b}, rethinking the core computation from Algorithm {a}: 10 -> 12 (regressed)" in prompt

    class Broken:
        def complete(self, *args, **kwargs):
            raise TimeoutError("down")

    after, calls = reflect_and_apply(tree, (region,), model=Broken(), problem=PROBLEM)
    assert calls[0]["status"] == "error"
    assert after.edges[c].contracted and after.edges[c].parent_id == a
    assert after.edges[c].experience == "Tried jumping from a."

    root_prune = SubtreePruning(0, (a,))
    root_prompt, _ = build_reflection_prompt(after, root_prune, PROBLEM)
    assert "designed from scratch" in root_prompt
