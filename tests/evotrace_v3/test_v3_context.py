from copy import deepcopy

from evotrace_v3 import Task
from evotrace_v3.context import build_evolution_context, evolution_prompt
from evotrace_v3.models import Edge, Node, TreePopulation


def add(tree, parent, fit, op="Refine"):
    i = max(tree.nodes, default=0) + 1
    tree.add(Node(i, f"def solve(x): return x + {i}", f"Compute offset {i}", fit),
             Edge(parent, i, op if parent else None), evaluations=1)
    return i


def test_crossover_reads_the_current_lineage_and_one_reference_card():
    tree = TreePopulation()
    a = add(tree, 0, 9)
    b = add(tree, a, 8)
    c = add(tree, b, 7, "Crossover")
    current = add(tree, c, 6)
    reference = add(tree, b, 5)
    tree.nodes[a].experience = "EARLY_EVIDENCE"
    tree.nodes[b].experience = "FORK_EVIDENCE"
    tree.edges[c].experience = "SHIFT_EVIDENCE"
    tree.edges[reference].experience = "REF_STAGE"
    tree.nodes[reference].experience = "REF_ATTEMPTS"
    tree.root_experience = "INITIAL_EVIDENCE"
    before = deepcopy(tree)
    text, instruction = build_evolution_context(tree, current, "Refine")
    for evidence in ["EARLY_EVIDENCE", "FORK_EVIDENCE", "SHIFT_EVIDENCE"]:
        assert evidence in text
    assert "INITIAL_EVIDENCE" not in text
    assert "Explore" not in instruction
    assert "Algorithm " not in text.replace("[Current Algorithm]", "")

    text, _ = build_evolution_context(tree, current, "Crossover", reference)
    assert (text.index("[Development of the Current Algorithm]") < text.index("[Other Algorithm]")
            < text.index("[Current Algorithm]"))
    assert "Common Ancestor" not in text and text.count("```python") == 2 and text.rstrip().endswith("```")
    for evidence in ["EARLY_EVIDENCE", "FORK_EVIDENCE", "SHIFT_EVIDENCE", "REF_STAGE", "REF_ATTEMPTS"]:
        assert text.count(evidence) == 1
    assert "INITIAL_EVIDENCE" not in text
    assert text.count(f"Compute offset {b}") == 1 and text.count(f"Compute offset {reference}") == 1

    # A reference that is a direct child of the current algorithm appears only as the reference.
    text, _ = build_evolution_context(tree, b, "Crossover", c)
    assert text.count(f"Compute offset {c}") == 1 and f"Compute offset {reference}" in text

    muted, _ = build_evolution_context(tree, current, "Crossover", reference, include_experiences=False)
    assert "FORK_EVIDENCE" not in muted and "REF_STAGE" not in muted
    assert f"return x + {current}" in muted and f"return x + {reference}" in muted
    assert tree == before


def test_explore_shows_cards_without_code_or_local_lineage_experiences():
    tree = TreePopulation()
    a = add(tree, 0, 9)
    branch = add(tree, a, 8)
    other = add(tree, 0, 7)
    tree.root_experience = "ROOT_NOTE"
    tree.nodes[a].experience = "ANCESTOR_NOTE"
    tree.nodes[branch].experience = "BRANCH_NOTE"
    tree.edges[other].experience = "OTHER_EDGE_NOTE"
    tree.nodes[other].experience = "OTHER_NOTE"
    task = Task("Minimize", "def solve(x): pass", lambda s, f: f(1), 2)
    prompt = evolution_prompt(task, tree, 0, "Explore", explore_ids=(branch, other))
    assert prompt.index(f"Compute offset {other}") < prompt.index(f"Compute offset {branch}")
    assert prompt.count("ROOT_NOTE") == 1
    for local_note in ["ANCESTOR_NOTE", "BRANCH_NOTE", "OTHER_EDGE_NOTE", "OTHER_NOTE"]:
        assert local_note not in prompt
    assert f"return x + {branch}" not in prompt and f"return x + {other}" not in prompt
    muted, _ = build_evolution_context(tree, 0, "Explore", explore_ids=(branch,), include_experiences=False)
    assert f"Compute offset {branch}" in muted and "ROOT_NOTE" not in muted


def test_refine_gives_designs_and_experience_and_code_only_for_the_current_algorithm():
    tree = TreePopulation()
    a = add(tree, 0, 9)
    b = add(tree, a, 8)
    parent = add(tree, b, 7)
    tried = add(tree, parent, 6)
    failed = max(tree.nodes) + 1
    tree.add(Node(failed, "def solve(x): return x / 0", "Divide by zero", None, failure="ZeroDivisionError"),
             Edge(parent, failed, "Refine"), evaluations=1)
    tree.edges[b].contracted, tree.edges[b].operator = True, None
    tree.edges[b].experience = "STAGE_NOTE"
    tree.nodes[b].experience = "B_NOTE"
    tree.nodes[parent].experience = "PARENT_NOTE"
    task = Task("Minimize", "def solve(x): pass", lambda s, f: f(1), 2)
    prompt = evolution_prompt(task, tree, parent, "Refine")
    text = prompt[prompt.index("[Context]"):prompt.index("[Instruction]")]
    assert (text.index("[Development of the Current Algorithm]") < text.index("[Attempts from the Current")
            < text.index("[Current Algorithm]"))
    assert text.rstrip().endswith("```") and text.count("```python") == 1 and "```diff" not in text
    assert f"return x + {a}" not in text and f"return x + {tried}" not in text and "x / 0" not in text
    assert "STAGE_NOTE" in text and "B_NOTE" in text and text.count("PARENT_NOTE") == 1
    assert "ZeroDivisionError" in text
    for design in [f"Compute offset {a}", f"Compute offset {b}", f"Compute offset {tried}", "Divide by zero"]:
        assert design in text
