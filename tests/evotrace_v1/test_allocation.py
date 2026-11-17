import math
import random

import pytest

from evotrace_v1.allocation import (
    AllocationConfig,
    allocate_step,
    eligible_reference_pool,
    parent_probabilities,
    select_crossover_reference,
)
from evotrace_v1.models import Edge, Node, TreePopulation
from evotrace_v1.topology import TopologyConfig, apply_consolidation, plan_consolidation


def _code(kind: str, c: float = 1.0) -> str:
    if kind == "linear":
        return f"def solve(x):\n    return x * {c} + 1.0\n"
    if kind == "quad":
        return f"def solve(x):\n    return (x ** 2) / (1.0 + abs(x)) * {c}\n"
    if kind == "loop":
        return (
            "def solve(x):\n"
            "    s = 0.0\n"
            f"    for i in range(5):\n        s += (x + i) * {c}\n"
            "    return s\n"
        )
    return f"def solve(x):\n    return max(x, {c}) - min(x, 0.2)\n"


def test_shinka_weighted_probabilities_persist_children_count_across_consolidation():
    tree = TreePopulation()
    tree.add(Node(1, _code("linear"), "d1", 6.0), Edge(0, 1, None), evaluations=1)
    tree.add(Node(2, _code("quad"), "d2", 6.0), Edge(0, 2, None), evaluations=1)
    tree.add(Node(3, _code("loop"), "d3", 8.0), Edge(0, 3, None), evaluations=1)
    tree.add(Node(4, _code("other"), "d4", 10.0), Edge(0, 4, None), evaluations=1)

    tree.add(Node(5, _code("linear", 2.0), "bad1", 20.0), Edge(1, 5, "Refine"), evaluations=1)
    tree.add(Node(6, _code("linear", 3.0), "bad2", 25.0), Edge(1, 6, "Refine"), evaluations=1)
    tree.add(Node(7, "bad (", "err", None, failure="SyntaxError"), Edge(1, 7, "Refine"), evaluations=1)

    assert tree.nodes[1].children_count == 3 and tree.nodes[2].children_count == 0
    cfg = TopologyConfig(base_capacity=4, trigger_capacity=6)
    rebuilt = apply_consolidation(tree, plan_consolidation(tree, cfg))
    assert set(rebuilt.nodes) == {1, 2, 3, 4}
    assert rebuilt.nodes[1].children_count == 3

    probs = parent_probabilities(rebuilt)
    assert math.isclose(sum(probs.values()), 1.0)
    assert math.isclose(probs[2] / probs[1], 4.0, rel_tol=1e-9)
    assert probs[2] > probs[3] > probs[4]


def test_reference_pool_keeps_all_when_small_and_filters_tail_when_large():
    small = TreePopulation()
    for i in range(1, 9):
        small.add(Node(i, _code("linear", i), f"d{i}", float(i)), Edge(0, i, None), evaluations=1)
    assert len(eligible_reference_pool(small, 1)) == 7

    large = TreePopulation()
    for i in range(1, 22):
        large.add(Node(i, _code("linear", i), f"d{i}", float(i)), Edge(0, i, None), evaluations=1)
    pool = eligible_reference_pool(large, 1)
    assert 21 not in pool and 20 not in pool and len(pool) == 14


def test_crossover_selects_most_dissimilar_and_avoids_repeating_used_pair():
    tree = TreePopulation()
    tree.add(Node(1, _code("linear", 1.0), "p1", 5.0), Edge(0, 1, None), evaluations=1)
    tree.add(Node(2, _code("linear", 2.0), "clone", 9.0), Edge(0, 2, None), evaluations=1)
    tree.add(Node(3, _code("quad", 1.0), "mid", 5.2), Edge(0, 3, None), evaluations=1)
    tree.add(Node(4, _code("loop", 1.0), "far", 5.1), Edge(0, 4, None), evaluations=1)

    assert select_crossover_reference(tree, 1) == 4
    tree.add(Node(5, _code("linear", 3.0), "cross_child", 12.0), Edge(1, 5, "Crossover", reference_id=4), evaluations=1)
    assert select_crossover_reference(tree, 1) == 3

    cfg = TopologyConfig(base_capacity=4, trigger_capacity=5)
    rebuilt = apply_consolidation(tree, plan_consolidation(tree, cfg))
    assert (1, 4) in rebuilt.used_crossover_pairs


def test_allocate_step_falls_back_to_refine_when_all_candidates_have_identical_normalized_source():
    tree = TreePopulation()
    tree.add(Node(1, 'def solve(x):\n    """Doc A"""\n    return x * 1.0 + 1.0\n', "d1", 5.0), Edge(0, 1, None), evaluations=1)
    tree.add(Node(2, 'def solve(x):\n    """Doc B"""\n    return x * 1.0 + 1.0\n', "d2", 6.0), Edge(0, 2, None), evaluations=1)

    class ForceCrossover(random.Random):
        def choices(self, population, weights=None, *, cum_weights=None, k=1):
            return ["Crossover"] if "Crossover" in population else [population[0]]

    _, operator, ref_id = allocate_step(tree, ForceCrossover())
    assert operator == "Refine" and ref_id is None


def test_invalid_allocation_config_rejected():
    with pytest.raises(ValueError, match="parent_selection_lambda"):
        AllocationConfig(parent_selection_lambda=-1.0)
