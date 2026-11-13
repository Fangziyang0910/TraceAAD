import random

import pytest

from evotrace_v1.models import Edge, Node, TreePopulation
from evotrace_v1.topology import TopologyConfig, code_distance, consolidate_topology, needs_consolidation, plan_consolidation


def _make_code(kind: str, coeff: float = 1.0) -> str:
    if kind == "linear":
        return f"def solve(x):\n    return x * {coeff} + 1.0\n"
    if kind == "quad":
        return f"def solve(x):\n    val = (x ** 2) * {coeff}\n    return val / (1.0 + abs(x))\n"
    if kind == "loop":
        return (
            "def solve(x):\n"
            "    acc = 0.0\n"
            f"    for i in range(5):\n        acc += (x + i) * {coeff}\n"
            "    return acc\n"
        )
    if kind == "branch":
        return (
            "def solve(x):\n"
            f"    if x > {coeff}:\n        return x * 0.5\n"
            "    return min(x, 0.0) - 1.0\n"
        )
    return f"def solve(x):\n    return max(x, {coeff}) - min(x, 0.1) * {coeff}\n"


def test_code_distance_is_invariant_to_numeric_constants_and_docstrings():
    c1 = 'def solve(x):\n    """Doc A."""\n    return x * 1.25 + 0.5\n'
    c2 = 'def solve(x):\n    """Doc B."""\n    return x * 9.80 + 3.1\n'
    c3 = _make_code("loop", 2.0)
    assert code_distance(c1, c2) == 0.0
    assert code_distance(c1, c3) > 0.5


def test_consolidation_contracts_superseded_path_preserves_lca_and_partitions_all_removed_nodes():
    tree = TreePopulation(root_id=0)

    def add(nid, parent, op, fit, kind, coeff=1.0, ref=None):
        node = Node(
            id=nid,
            code=_make_code(kind, coeff),
            design=f"Design {nid}",
            fitness=fit,
            failure="runtime error" if fit is None else None,
        )
        tree.add(node, Edge(parent, nid, op, ref), evaluations=1)

    # Direction 1 under #1: chain 1 -> 2 -> 3 -> 4 (where 2, 3 are minor linear tweaks superseded by 4)
    # and a second active branch 1 -> 5 -> 6 so #1 must be retained as LCA!
    add(1, 0, None, 10.0, "linear", 1.0)
    add(2, 1, "Refine", 9.5, "linear", 1.1)
    add(3, 2, "Crossover", 9.2, "linear", 1.2, ref=10)
    add(4, 3, "Refine", 7.0, "quad", 1.0)  # Best in sub-branch A

    add(5, 1, "Refine", 9.8, "branch", 1.0)
    add(6, 5, "Refine", 7.2, "branch", 2.0)  # Best in sub-branch B

    # Dead side branches & infeasible leaf under #1 and #3
    add(7, 1, "Refine", 15.0, "linear", 3.0)
    add(8, 1, "Refine", None, "linear", 4.0)
    add(9, 3, "Refine", 14.0, "linear", 5.0)

    # Direction 2 via Explore #10, and a young Explore #11
    add(10, 0, None, 8.0, "loop", 1.0)
    add(11, 4, "Explore", 11.0, "other", 1.0)

    cfg = TopologyConfig(
        base_capacity=6,
        trigger_capacity=10,
        min_protected_evaluations=3,
    )
    assert needs_consolidation(tree, cfg)

    new_tree, plan = consolidate_topology(tree, cfg)

    # Invariants check
    assert len(new_tree.nodes) <= cfg.base_capacity
    assert all(n.fitness is not None for n in new_tree.nodes.values())
    assert set(new_tree.nodes) == set(new_tree.edges) == set(plan.kept_ids)

    # Global best #4, second branch best #6, LCA #1, Dir2 #10, and protected young Explore #11 should survive
    assert {1, 4, 6, 10, 11}.issubset(new_tree.nodes)
    assert 1 in plan.lca_anchor_ids

    # Intermediate nodes [2, 3] between 1 and 4 are contracted into edge 1 -> 4 with merged operator Crossover
    assert new_tree.edges[4].parent_id == 1
    assert new_tree.edges[4].operator == "Crossover"
    assert new_tree.edges[4].reference_id == 10
    contr_4 = next(c for c in plan.path_contractions if c.child_id == 4)
    assert contr_4.parent_id == 1 and contr_4.removed_ids == (2, 3)

    # Exact accounting: every node in original tree is either kept, in a path contraction, or in a branch consolidation
    accounted_removed = set()
    for c in plan.path_contractions:
        assert not (accounted_removed & set(c.removed_ids))
        accounted_removed.update(c.removed_ids)
    for b in plan.branch_consolidations:
        assert not (accounted_removed & set(b.removed_ids))
        accounted_removed.update(b.removed_ids)

    assert accounted_removed | set(plan.kept_ids) == set(tree.nodes.keys())
    assert not (accounted_removed & set(plan.kept_ids))


def test_consolidation_contracts_old_direction_origin_and_keeps_actual_evaluation_count():
    tree = TreePopulation(root_id=0)
    tree.add(Node(1, _make_code("linear"), "d1", 9.0), Edge(0, 1, None), evaluations=1)
    tree.add(Node(2, _make_code("quad"), "d2", 9.5), Edge(1, 2, "Explore"), evaluations=1)
    tree.add(Node(3, _make_code("quad", 2.0), "d3", 8.5), Edge(2, 3, "Refine"), evaluations=1)
    tree.add(Node(4, _make_code("loop"), "d4", 7.5), Edge(3, 4, "Refine"), evaluations=1)
    for extra in range(5, 10):
        tree.add(Node(extra, _make_code("branch", extra), f"d{extra}", 20.0 + extra),
                 Edge(1, extra, "Refine"), evaluations=1)

    cfg = TopologyConfig(base_capacity=2, trigger_capacity=8)
    new_tree, plan = consolidate_topology(tree, cfg)

    assert set(new_tree.nodes) == {1, 4}
    assert new_tree.edges[4].parent_id == 1 and new_tree.edges[4].operator == "Explore"
    assert plan.path_contractions[0].removed_ids == (2, 3)
    assert new_tree.direction_of(4) == 2
    assert new_tree.direction_evaluations[2] == 3
    new_tree.add(Node(10, _make_code("branch"), "next", 7.0), Edge(4, 10, "Refine"), evaluations=2)
    again, plan = consolidate_topology(new_tree, cfg)
    assert set(again.nodes) == {1, 10}
    assert again.direction_of(10) == 2
    assert again.direction_evaluations[2] == 5
    assert plan.protected_ids == ()


def test_when_feasible_within_base_capacity_only_infeasible_leaves_are_consolidated():
    tree = TreePopulation(root_id=0)
    for nid in range(1, 4):
        tree.add(Node(nid, _make_code("linear", nid), f"d{nid}", float(nid)),
                 Edge(0, nid, None), evaluations=1)
    tree.add(Node(4, "bad syntax (", "fail", None, failure="SyntaxError"),
             Edge(1, 4, "Refine"), evaluations=1)

    plan = plan_consolidation(tree, TopologyConfig(base_capacity=5, trigger_capacity=6))
    assert plan.kept_ids == (1, 2, 3)
    assert plan.path_contractions == ()
    assert len(plan.branch_consolidations) == 1
    assert plan.branch_consolidations[0].parent_id == 1
    assert plan.branch_consolidations[0].removed_ids == (4,)


def test_invalid_topology_config_rejected():
    with pytest.raises(ValueError, match="base_capacity"):
        TopologyConfig(base_capacity=1, trigger_capacity=5)
    with pytest.raises(ValueError, match="trigger_capacity"):
        TopologyConfig(base_capacity=10, trigger_capacity=10)


def test_internal_version_competes_even_when_its_descendant_is_better():
    tree = TreePopulation()
    for i, parent, fit, kind in [(1, 0, 6, "linear"), (2, 1, 2, "loop"), (3, 2, 1, "linear"),
                                 (4, 1, 4, "linear"), (5, 1, 5, "linear")]:
        tree.add(Node(i, _make_code(kind), "", fit), Edge(parent, i, None if parent == 0 else "Refine"), evaluations=1)
    plan = plan_consolidation(tree, TopologyConfig(base_capacity=3, trigger_capacity=6))
    assert plan.kept_ids == (1, 2, 3)
    assert 2 in plan.selected_ids


@pytest.mark.parametrize("capacity", [2, 32])
def test_long_explore_history_fits_capacity_while_retaining_champion(capacity):
    tree = TreePopulation()
    for nid in range(1, 6):
        tree.add(Node(nid, _make_code("linear"), "initial", 100.0 + nid),
                 Edge(0, nid, None), evaluations=1)
    parent = 1
    for nid in range(6, 38):
        tree.add(Node(nid, _make_code("linear"), "path", 37.0 - nid),
                 Edge(parent, nid, "Explore" if nid % 2 else "Refine"), evaluations=1)
        parent = nid
    for nid in range(38, 65):
        tree.add(Node(nid, _make_code("linear"), "side", 1000.0 + nid),
                 Edge(2, nid, "Refine"), evaluations=1)

    rebuilt, plan = consolidate_topology(tree, TopologyConfig(base_capacity=capacity))
    assert 37 in rebuilt.nodes and len(rebuilt.nodes) <= capacity
    assert rebuilt.edges[37].parent_id == 1
    assert rebuilt.edges[37].operator == "Explore"
    assert len(plan.path_contractions[0].removed_ids) == 31
    assert len(tree.nodes) == 64
    # The rebuilt tree remains valid for the next consolidation.
    plan_consolidation(rebuilt, TopologyConfig(base_capacity=capacity))


def test_internal_fork_is_retained_while_historical_direction_endpoints_contract():
    tree = TreePopulation()
    rows = [(1, 0, None, 10, "linear"), (2, 1, "Refine", 9, "linear"),
            (3, 2, "Explore", 8, "linear"), (4, 3, "Refine", 7, "linear"),
            (5, 4, "Refine", 1, "loop"), (6, 4, "Refine", 2, "branch"),
            (7, 1, "Refine", 20, "linear")]
    for i, parent, op, fit, kind in rows:
        tree.add(Node(i, _make_code(kind), "", fit), Edge(parent, i, op), evaluations=1)
    plan = plan_consolidation(tree, TopologyConfig(base_capacity=6, trigger_capacity=8))
    assert plan.selected_ids == (5, 6)
    assert plan.kept_ids == (1, 4, 5, 6)
    assert plan.direction_anchor_ids == (1,)
    assert plan.lca_anchor_ids == (4,)


def test_evaluation_counts_and_latest_id_survive_pruning_without_reprotecting_old_direction():
    tree = TreePopulation()
    for i, parent, op, fit, cost in [(1, 0, None, 10, 1), (2, 1, "Explore", 1, 2),
                                    (3, 2, "Refine", 20, 2), (4, 2, "Explore", 11, 1),
                                    (5, 1, "Refine", 30, 1)]:
        tree.add(Node(i, _make_code("linear"), "", fit), Edge(parent, i, op), evaluations=cost)
    assert tree.direction_evaluations == {1: 2, 2: 4, 4: 1}
    cfg = TopologyConfig(base_capacity=3, trigger_capacity=6)
    first, plan = consolidate_topology(tree, cfg)
    assert plan.protected_ids == (4,)
    assert set(first.nodes) == {1, 2, 4}
    assert first.direction_evaluations == tree.direction_evaluations
    assert first.latest_node_id == 5
    with pytest.raises(ValueError, match="increase"):
        first.add(Node(5, _make_code("linear"), "", 0), Edge(2, 5, "Refine"), evaluations=1)
    first.add(Node(6, _make_code("quad"), "", 0.5), Edge(2, 6, "Refine"), evaluations=1)
    second, plan = consolidate_topology(first, cfg)
    assert plan.protected_ids == ()
    assert plan.protection_capacity_skips == (4,)
    assert second.direction_evaluations == {1: 2, 2: 5}
    assert second.latest_node_id == 6
    assert tree.direction_evaluations == {1: 2, 2: 4, 4: 1}


@pytest.mark.parametrize("origin_operator", ["Refine", "Explore"])
def test_contracted_crossover_keeps_latest_actual_reference_even_when_it_was_removed(origin_operator):
    tree = TreePopulation()
    rows = [(1, 0, None, 100, "linear", None), (2, 0, None, 0, "branch", None),
            (3, 1, origin_operator, 80, "linear", None), (4, 3, "Crossover", 70, "linear", 2),
            (5, 4, "Crossover", 60, "linear", 3), (6, 5, "Refine", 1, "loop", None)]
    for i, parent, op, fit, kind, ref in rows:
        tree.add(Node(i, _make_code(kind), "", fit), Edge(parent, i, op, ref), evaluations=1)
    rebuilt, plan = consolidate_topology(tree, TopologyConfig(base_capacity=3, trigger_capacity=6))
    assert plan.kept_ids == (1, 2, 6)
    expected_operator = "Explore" if origin_operator == "Explore" else "Crossover"
    assert rebuilt.edges[6].parent_id == 1 and rebuilt.edges[6].operator == expected_operator
    assert rebuilt.edges[6].reference_id == 3
    rebuilt.add(Node(7, _make_code("loop"), "next", 0.5), Edge(6, 7, "Refine"), evaluations=1)
    again, _ = consolidate_topology(rebuilt, TopologyConfig(base_capacity=3, trigger_capacity=6))
    assert again.edges[7].operator == expected_operator
    assert again.edges[7].reference_id == 3


def test_repeated_growth_and_consolidation_preserve_capacity_partition_and_direction_counts():
    rng = random.Random(17)
    tree = TreePopulation()
    actual_evaluations = {}
    best_fitness = float("inf")
    cycles = 0
    for nid in range(1, 257):
        parent = 0 if nid <= 5 else rng.choice(list(tree.nodes))
        op = None if parent == 0 else rng.choice(["Refine", "Crossover", "Explore"])
        reference = rng.choice(list(tree.nodes)) if op == "Crossover" else None
        cost = rng.choice([1, 2])
        fitness = rng.random() * 100
        best_fitness = min(best_fitness, fitness)
        tree.add(Node(nid, _make_code(rng.choice(["linear", "quad", "loop", "branch"]), nid), "", fitness),
                 Edge(parent, nid, op, reference), evaluations=cost)
        direction = tree.direction_of(nid)
        actual_evaluations[direction] = actual_evaluations.get(direction, 0) + cost
        if needs_consolidation(tree):
            before = set(tree.nodes)
            tree, plan = consolidate_topology(tree)
            cycles += 1
            removed = [n for c in plan.path_contractions for n in c.removed_ids]
            removed += [n for b in plan.branch_consolidations for n in b.removed_ids]
            assert len(removed) == len(set(removed))
            assert set(removed) == before - set(tree.nodes)
            assert len(tree.nodes) <= 32
            assert min(n.fitness for n in tree.nodes.values()) == best_fitness
            assert tree.direction_evaluations == {
                d: actual_evaluations[d] for d in {tree.direction_of(n) for n in tree.nodes}
            }
            for protected in plan.protected_ids:
                origin = tree.direction_of(protected)
                assert origin > 5 and tree.latest_node_id - origin <= 16
                assert actual_evaluations[origin] < 4
            plan_consolidation(tree)
    assert cycles >= 3
