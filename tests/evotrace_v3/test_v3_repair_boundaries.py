import json

from evotrace_v3.models import Edge, Node, TreePopulation
from evotrace_v3.reflection import build_reflection_prompt, reflect_and_apply
from evotrace_v3.topology import PathContraction, SubtreePruning

PROBLEM = "[Problem]\nMinimize.\nScore: lower is better."


def add(tree, parent, score, code, op="Refine"):
    nid = max(tree.nodes, default=0) + 1
    root_attached = parent == 0 or op == "Explore"
    tree.add(
        Node(nid, code, f"Design {nid}", score),
        Edge(0 if root_attached else parent, nid, op if parent else None),
        evaluations=1,
    )
    return nid


class Reply:
    def __init__(self, text):
        self.content, self.usage = text, {"prompt_tokens": 1}


def fixed(text, prompts=None):
    class Model:
        def complete(self, prompt, **kwargs):
            if prompts is not None:
                prompts.append(prompt)
            return Reply(json.dumps({"experience": text}))
    return Model()


class Down:
    def complete(self, *args, **kwargs):
        raise TimeoutError("endpoint down")


def test_negative_scores_are_compared_by_program_and_persisted_on_the_edge():
    tree = TreePopulation()
    start = add(tree, 0, -14.848125, "def solve(x): return x")
    middle = add(tree, start, -14.696875, "def solve(x): return x + 1")
    end = add(tree, middle, -11.508125, "def solve(x): return x * 2", "Crossover")
    region = PathContraction(start, end, (middle,))
    prompt, _ = build_reflection_prompt(tree, region, PROBLEM)
    assert "overall -14.8481 -> -11.5081 (regressed)" in prompt
    assert f"Algorithm {middle}, modified from Algorithm {start}: -14.8481 -> -14.6969 (regressed)" in prompt
    assert (f"Algorithm {end} (kept), combining Algorithm {middle} with another algorithm: -14.6969 -> -11.5081 "
            "(regressed)") in prompt
    new, calls = reflect_and_apply(tree, (region,), model=fixed("Identity to doubling."), problem=PROBLEM)
    assert calls[0]["error"] is None and calls[0]["usage"] == {"prompt_tokens": 1}
    assert new.edges[end].experience == "Identity to doubling."
    assert new.edges[end].contracted and new.edges[end].parent_id == start


def test_representative_cover_excludes_worse_nearby_offspring_and_resets_direct_effort_on_novel_or_improving_child():
    import pytest
    from evotrace_v3.allocation import parent_probabilities
    from evotrace_v3.topology import representatives

    t = TreePopulation()
    a = add(t, 0, 2.0, "def f(x): return x + 2")
    b = add(t, a, 1.0, "def f(x): return x + 1")
    c = add(t, 0, 3.0, "def f(x): return [y * y for y in range(x) if y % 2 == 0]")
    assert representatives(t) == (b, c)
    before = parent_probabilities(t)
    assert set(before) == {b, c}
    add(t, b, None, "def f(x): raise ValueError()")
    after = parent_probabilities(t)
    assert after[b] / after[c] == pytest.approx(before[b] / before[c] / 2)
    worse_tweak = add(t, b, 1.5, "def f(x): return x + 5")
    assert worse_tweak not in parent_probabilities(t)
    novel = add(t, b, 2.5, "def f(x):\n    acc = 0\n    for k in range(x):\n        acc ^= (k << 1)\n    return acc", "Explore")
    probs = parent_probabilities(t)
    assert set(probs) == {b, novel, c}
    assert t.nodes[novel].direct_evaluations == 0
    assert probs[novel] > probs[c] > 0


def test_each_algorithm_appears_once_with_design_and_code_and_no_diff():
    t = TreePopulation()
    a = add(t, 0, 10, "def f(): return 0")
    b = add(t, a, 11, "def f(): return 1")
    c = add(t, b, 8, "def f(): return 2")
    x = add(t, b, 12, "def f(): return 3")
    prompt, _ = build_reflection_prompt(t, PathContraction(a, c, (b,), (x,)), PROBLEM)
    for i in (a, b, c, x):
        assert prompt.count(f"Design: Design {i}") == 1
        assert prompt.count(f"return {i - 1}") == 1
    assert f"Algorithm {x}, modified from Algorithm {b}: 11 -> 12 (regressed)" in prompt
    assert "```diff" not in prompt and "[Related Results]" not in prompt and "Comparison" not in prompt


def test_new_experience_replaces_the_region_experience_and_prompt_size_follows_the_region():
    t = TreePopulation()
    a = add(t, 0, -10, "def f(): return 0")
    b = add(t, a, -8, "def f(): return 1")
    x = add(t, b, -9, "def f(): return 2")
    prompts = []
    model = fixed("Changed the return value.", prompts)
    t, _ = reflect_and_apply(t, (SubtreePruning(b, (x,)),), model=model, problem=PROBLEM)
    assert t.nodes[b].experience == "Changed the return value."
    t, _ = reflect_and_apply(t, (SubtreePruning(a, (b,)),), model=model, problem=PROBLEM)
    assert prompts[-1].count("Changed the return value.") == 1
    assert t.nodes[a].experience == "Changed the return value."
    lengths = []
    for step in range(12):
        k1 = add(t, a, -7 + step, f"def f(): return {10 + step}")
        k2 = add(t, a, -6 + step, f"def f(): return {30 + step}")
        t, _ = reflect_and_apply(t, (SubtreePruning(a, (k1, k2)),), model=model, problem=PROBLEM)
        lengths.append(len(prompts[-1]))
    assert max(lengths) - min(lengths) < 60


def test_failed_reflection_carries_the_region_experience_forward():
    t = TreePopulation()
    a = add(t, 0, 10, "def f(): return 0")
    b = add(t, a, 15, "def f(): return 1")
    c = add(t, b, 12, "def f(): return 2")
    t.edges[b].experience = "Stage note on b."
    t.nodes[b].experience = "Attempts at b."
    t.nodes[a].experience = "Attempts at a stay on a."
    region = PathContraction(a, c, (b,))
    new, calls = reflect_and_apply(t, (region,), model=fixed("  "), problem=PROBLEM)
    assert "empty" in calls[0]["error"]
    assert new.edges[c].experience == "Stage note on b. Attempts at b."
    assert new.nodes[a].experience == "Attempts at a stay on a."
    new, calls = reflect_and_apply(t, (region,), model=Down(), problem=PROBLEM)
    assert calls[0]["error"].startswith("TimeoutError") and new.edges[c].experience == "Stage note on b. Attempts at b."


def test_virtual_root_has_no_overall_outcome():
    t = TreePopulation()
    a = add(t, 0, 15, "def f(): return 1")
    b = add(t, a, 12, "def f(): return 2")
    prompt, _ = build_reflection_prompt(t, PathContraction(0, b, (a,)), PROBLEM)
    assert f"Algorithm {b} was developed along Algorithm {a} -> Algorithm {b}, starting from a design made from scratch" in prompt
    assert "overall" not in prompt
    assert f"Algorithm {a}, designed from scratch: fitness = 15" in prompt


def test_designs_from_scratch_are_listed_from_best_to_worst():
    t = TreePopulation()
    worse = add(t, 0, -9.4, "def f(): return 1")
    failed = add(t, 0, None, "def f(): raise ValueError()")
    better = add(t, 0, -13.2, "def f(): return 2")
    prompt, _ = build_reflection_prompt(t, SubtreePruning(0, (worse, failed, better)), PROBLEM)
    assert "listed from best to worst fitness" in prompt
    heads = [prompt.index(f"Algorithm {i}, designed from scratch") for i in (better, worse, failed)]
    assert heads == sorted(heads)


def test_outage_keeps_the_parent_note_while_pruning_proceeds():
    t = TreePopulation()
    a = add(t, 0, 10, "def f(): return 0")
    b = add(t, a, 8, "def f(): return 1")
    c = add(t, a, 12, "def f(): return 2")
    note = "The two changes offered different tradeoffs under the same starting computation."
    t, _ = reflect_and_apply(t, (SubtreePruning(a, (b, c)),), model=fixed(note), problem=PROBLEM)
    assert t.nodes[a].experience == note
    d = add(t, a, None, "def f(): raise ValueError()")
    region = SubtreePruning(a, (d,))
    prompt, _ = build_reflection_prompt(t, region, PROBLEM)
    assert note in prompt and "Failure:" in prompt
    t, calls = reflect_and_apply(t, (region,), model=Down(), problem=PROBLEM)
    assert calls[0]["error"] and t.nodes[a].experience == note and d not in t.nodes


def test_reflection_reads_only_the_region_and_its_anchors():
    from dataclasses import asdict

    t = TreePopulation()
    a = add(t, 0, 10, "def f(): return 0")
    b = add(t, a, 12, "def f(): return 1")
    better = add(t, a, 8, "def f(): return 12345")
    add(t, 0, 1, "def f(): return 54321")
    t.nodes[better].design = "Joint routing and capacity model"
    t.nodes[better].experience = "Notes on another branch stay on their owner."
    before = asdict(t)
    prompt, _ = build_reflection_prompt(t, SubtreePruning(a, (b,)), PROBLEM)
    assert f"Algorithm {a} (kept): fitness = 10" in prompt
    assert "12345" not in prompt and "54321" not in prompt
    assert t.nodes[better].design not in prompt and t.nodes[better].experience not in prompt
    assert asdict(t) == before
