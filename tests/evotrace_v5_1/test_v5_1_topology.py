"""有界谱系森林拓扑归约测试：就地升级、零死锁容量淘汰、级联继承与规范 AST 去重。"""

import random

from evotrace_v5_1.models import ROOT_ID, Evaluation, Program, TreePopulation
from evotrace_v5_1.topology import (
    apply_reduction,
    bare_code,
    code_identity,
    duplicate_of,
    next_reduction,
)


def make_prog(pid: int, fitness: float, *, parent_id: int = ROOT_ID, code: str | None = None) -> Program:
    src = code if code is not None else f"def solve(x):\n    return x + {pid}\n"
    return Program(pid, src, f"Design {pid}", "Refine", parent_id, None, Evaluation("success", fitness))


def test_in_place_upgrade_replaces_beaten_degree_one_chain_including_lineage_root():
    # v_0 -> 1 (10.0) -> 2 (12.0) -> 3 (8.0)
    archive = {1: make_prog(1, 10.0), 2: make_prog(2, 12.0, parent_id=1), 3: make_prog(3, 8.0, parent_id=2)}
    tree = TreePopulation(
        parents={1: ROOT_ID, 2: 1, 3: 2},
        idea_notes={1: "Idea from root 1"},
        direct_evals={1: 4, 2: 1, 3: 0},
    )
    event = next_reduction(tree, archive, capacity=24)
    assert event is not None
    assert event.kind == "upgrade"
    assert event.anchor == ROOT_ID
    assert event.target == 3
    assert event.removed == (1, 2)

    apply_reduction(tree, event, "Idea from root 1")
    assert tree.parents == {3: ROOT_ID}
    assert tree.idea_notes == {3: "Idea from root 1"}
    assert tree.direct_evals == {3: 0}
    assert tree.lineage_root(3) == 3
    assert next_reduction(tree, archive, capacity=24) is None


def test_zero_deadlock_on_adversarial_28_node_chain_and_fork():
    # Reproduces the exact Draft-1 Astra Max deadlock counterexample:
    # 4 single-node lineages + 1 deteriorating 27-node chain (1..27) ending in fork -> (28, champion 0).
    archive: dict[int, Program] = {}
    parents: dict[int, int] = {}
    for i in range(1, 5):
        archive[i] = make_prog(i, float(50 + i))
        parents[i] = ROOT_ID
    prev = ROOT_ID
    for step in range(1, 28):
        nid = 100 + step
        archive[nid] = make_prog(nid, float(step))
        parents[nid] = prev
        prev = nid
    archive[200] = make_prog(200, 28.0, parent_id=prev)
    parents[200] = prev
    archive[201] = make_prog(201, 0.0, parent_id=prev)
    parents[201] = prev

    tree = TreePopulation(parents=parents)
    steps = 0
    while (event := next_reduction(tree, archive, capacity=24)) is not None:
        apply_reduction(tree, event, "")
        steps += 1
        assert steps < 100
    assert len(tree.parents) <= 24
    assert 201 in tree.parents  # global best always preserved


def test_zero_deadlock_fuzz_over_200_random_forests_and_small_capacities():
    rng = random.Random(42)
    for trial in range(200):
        n_nodes = rng.randint(2, 40)
        capacity = rng.choice([1, 2, 5, 12, 24])
        archive: dict[int, Program] = {}
        parents: dict[int, int] = {}
        for nid in range(1, n_nodes + 1):
            # Mix distinct and tied fitness values
            fit = float(rng.randint(1, 8)) if trial % 4 == 0 else rng.uniform(0.0, 100.0)
            parent = ROOT_ID if (nid == 1 or rng.random() < 0.25) else rng.randint(1, nid - 1)
            archive[nid] = make_prog(nid, fit, parent_id=parent)
            parents[nid] = parent
        champion = min(parents, key=lambda i: (archive[i].fitness, i))
        tree = TreePopulation(parents=parents)
        guard = 0
        while (event := next_reduction(tree, archive, capacity=capacity)) is not None:
            apply_reduction(tree, event, "")
            guard += 1
            assert guard <= n_nodes + 5
        assert len(tree.parents) <= capacity
        assert champion in tree.parents
        assert ROOT_ID not in tree.idea_notes
        # Verify valid acyclic forest
        for nid in tree.parents:
            path = tree.path_to(nid)
            assert path and tree.parents[path[0]] == ROOT_ID


def test_evict_leaf_cascade_hands_fresh_note_to_winning_sibling():
    # Tree of size 4 with capacity=3:
    # v_0 -> 1(10.0) -> {2(8.0), 3(15.0)}, plus v_0 -> 4(9.0)
    # 1 has degree 2 so no immediate upgrade. Capacity=3 triggers evict_leaf on 3(15.0) under parent 1.
    # Once 3 is removed and note written to 1, 1 drops to degree 1 with child 2(8.0) > 1(10.0),
    # immediately triggering upgrade(1 -> 2) which hands note(1) straight to 2!
    archive = {
        1: make_prog(1, 10.0),
        2: make_prog(2, 8.0, parent_id=1),
        3: make_prog(3, 15.0, parent_id=1),
        4: make_prog(4, 9.0),
    }
    tree = TreePopulation(parents={1: ROOT_ID, 2: 1, 3: 1, 4: ROOT_ID})

    ev1 = next_reduction(tree, archive, capacity=3)
    assert ev1 is not None and ev1.kind == "evict_leaf" and ev1.target == 1 and ev1.removed == (3,)
    apply_reduction(tree, ev1, "Salvaged idea from leaf 3")
    assert tree.idea_notes[1] == "Salvaged idea from leaf 3"

    ev2 = next_reduction(tree, archive, capacity=3)
    assert ev2 is not None and ev2.kind == "upgrade" and ev2.target == 2 and ev2.removed == (1,)
    apply_reduction(tree, ev2, tree.idea_notes[1])
    assert tree.parents == {2: ROOT_ID, 4: ROOT_ID}
    assert tree.idea_notes == {2: "Salvaged idea from leaf 3"}
    assert next_reduction(tree, archive, capacity=3) is None


def test_retire_root_never_writes_note_to_virtual_root_and_evict_single_child_reconnects_fork():
    # Case 1: worst node is a single-node lineage root -> retire_root
    archive = {1: make_prog(1, 5.0), 2: make_prog(2, 20.0)}
    tree = TreePopulation(parents={1: ROOT_ID, 2: ROOT_ID}, idea_notes={2: "Old note"})
    ev = next_reduction(tree, archive, capacity=1)
    assert ev is not None and ev.kind == "retire_root" and ev.removed == (2,)
    apply_reduction(tree, ev, "Must be ignored")
    assert tree.parents == {1: ROOT_ID}
    assert ROOT_ID not in tree.idea_notes and 2 not in tree.idea_notes

    # Case 2: u(10) -> fork b(12) -> {c(8), d(9)} with capacity=3 -> evict_single_child on u(10)
    archive2 = {
        1: make_prog(1, 10.0),
        2: make_prog(2, 12.0, parent_id=1),
        3: make_prog(3, 8.0, parent_id=2),
        4: make_prog(4, 9.0, parent_id=2),
    }
    tree2 = TreePopulation(parents={1: ROOT_ID, 2: 1, 3: 2, 4: 2}, idea_notes={1: "Note on 1"})
    ev2 = next_reduction(tree2, archive2, capacity=3)
    assert ev2 is not None and ev2.kind == "evict_single_child" and ev2.target == 2 and ev2.removed == (1,)
    apply_reduction(tree2, ev2, "Note on 1")
    assert tree2.parents == {2: ROOT_ID, 3: 2, 4: 2}
    assert tree2.idea_notes == {2: "Note on 1"}


def test_bare_code_preserves_formatting_and_canonical_ast_dedup_allows_neutral_variants():
    raw = 'def solve(x):\n    """Docstring."""\n    # comment\n    return float(min(i for i in range(x)))\n'
    cleaned = bare_code(raw)
    assert '"""' not in cleaned and "#" not in cleaned
    # Must NOT introduce ast.unparse double-parenthesis bug `min((i for ...))`
    assert "min(i for i in range(x))" in cleaned

    p1 = make_prog(1, 10.0, code="def solve(x):\n    return x + 1\n")
    p2_dup = make_prog(2, 10.0, code='def solve(x):\n    """Same AST."""\n    return x + 1  # hi\n')
    p3_neutral = make_prog(3, 10.0, code="def solve(x):\n    y = x + 1\n    return y\n")
    assert code_identity(p1.code) == code_identity(p2_dup.code)
    assert duplicate_of(p2_dup, [p1]) == 1
    # Different code with identical scalar score MUST NOT be blocked (preserves neutral drift)
    assert duplicate_of(p3_neutral, [p1]) is None
