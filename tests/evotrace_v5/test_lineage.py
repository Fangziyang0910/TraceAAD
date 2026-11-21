"""无状态局部归约、非平稳后验预算分配与树谱系选供体/代表单元测试。"""

import random

import pytest

from evotrace_v5.allocation import (
    RECENCY_DECAY,
    History,
    Ledger,
    _select_donor,
    _select_explore_references,
    choose_direction,
    choose_operator,
    choose_parent,
    tree_distance,
)
from evotrace_v5.models import ROOT_ID, Attempt, Evaluation, Program, TreePopulation
from evotrace_v5.topology import (
    Reduction,
    apply_reduction,
    beats,
    code_identity,
    duplicate_of,
    reduction_plan,
)


def make_prog(pid, fitness, code=None, parent_id=None, duplicate_of_id=None):
    return Program(
        id=pid,
        code=code or f"def solve(x):\n    return x + {pid}\n",
        design=f"design {pid}",
        operator="Refine",
        parent_id=parent_id,
        reference_id=None,
        evaluation=Evaluation("success", fitness=fitness),
        duplicate_of=duplicate_of_id,
    )


def test_code_identity_ignores_comments_docstrings_and_formatting():
    a = 'def solve(x):\n    """doc"""\n    # comment\n    return x + 1\n'
    b = "def solve(x):\n    return (x+1)\n"
    c = "def solve(x):\n    return x + 2\n"
    assert code_identity(a) == code_identity(b)
    assert code_identity(a) != code_identity(c)

    p1 = make_prog(1, 7.0, a)
    p2 = make_prog(2, 7.0, b)
    p3 = make_prog(3, 7.0, c)
    assert duplicate_of(p2, [p1]) == 1
    assert duplicate_of(p3, [p1]) is None


def test_single_weak_leaf_stays_and_valley_recovery_contracts():
    archive = {1: make_prog(1, 6.4), 2: make_prog(2, 6.9), 3: make_prog(3, 7.1)}
    tree = TreePopulation(parents={1: ROOT_ID, 2: 1, 3: 2})
    assert reduction_plan(tree, archive) == []

    archive[4] = make_prog(4, 6.6)
    tree.parents[4] = 3
    assert reduction_plan(tree, archive) == [Reduction("path", 1, 4, (2, 3))]

    archive[2] = make_prog(2, 6.5)
    assert reduction_plan(tree, archive) == [Reduction("path", 2, 4, (3,))]


def test_monotonic_chain_contracts_and_live_fork_is_boundary():
    archive = {
        1: make_prog(1, 8.0),
        2: make_prog(2, 7.5),
        3: make_prog(3, 7.0),
        4: make_prog(4, 6.5),
    }
    tree = TreePopulation(parents={1: ROOT_ID, 2: 1, 3: 2, 4: 3})
    assert reduction_plan(tree, archive) == [Reduction("path", 1, 4, (2, 3))]

    archive[5] = make_prog(5, 7.4)
    tree.parents[5] = 2
    assert reduction_plan(tree, archive) == [Reduction("path", 2, 4, (3,))]


def test_stateless_pruning_keeps_parent_with_fresh_experience_and_is_idempotent():
    # 1 (7.2) -> 2 (7.7) -> {3 (8.1), 4 (8.3)}:
    # Stateless pruning retracts {3, 4} into 2 so 2 keeps its newly distilled experience as a single leaf.
    archive = {
        1: make_prog(1, 7.2),
        2: make_prog(2, 7.7),
        3: make_prog(3, 8.1),
        4: make_prog(4, 8.3),
    }
    tree = TreePopulation(parents={1: ROOT_ID, 2: 1, 3: 2, 4: 2})
    events = reduction_plan(tree, archive)
    assert events == [Reduction("branch", 2, 2, (3, 4))]

    for event in events:
        apply_reduction(tree, event, "learned from 3 and 4")
    assert tree.parents == {1: ROOT_ID, 2: 1}
    assert tree.node_notes == {2: "learned from 3 and 4"}
    assert reduction_plan(tree, archive) == []  # strict idempotence

    # Once 1 grows a second weak leaf 5 (7.9), {2, 5} retract cleanly into 1 together.
    archive[5] = make_prog(5, 7.9)
    tree.parents[5] = 1
    assert reduction_plan(tree, archive) == [Reduction("branch", 1, 1, (2, 5))]


def test_contraction_immediately_triggers_pruning_on_new_neighbors():
    archive = {
        1: make_prog(1, 7.2),
        2: make_prog(2, 8.1),
        3: make_prog(3, 7.6),
        4: make_prog(4, 7.5),
    }
    tree = TreePopulation(parents={1: ROOT_ID, 2: 1, 3: 2, 4: 1})
    assert reduction_plan(tree, archive) == [
        Reduction("path", 1, 3, (2,)),
        Reduction("branch", 1, 1, (3, 4)),
    ]


def test_five_evaluation_allocation_and_recency_decay():
    archive = {
        1: make_prog(1, 8.0),
        2: make_prog(2, 8.5, parent_id=1),
        3: make_prog(3, 7.9, parent_id=1),
        4: make_prog(4, 8.1, parent_id=2),
        5: make_prog(5, 7.9, parent_id=3, duplicate_of_id=3),
        6: make_prog(6, 7.9, parent_id=3, duplicate_of_id=3),
    }
    tree = TreePopulation(parents={1: ROOT_ID})
    ledger = Ledger()
    ledger.add(archive[1])

    def record(aid, parent, pid, eid, attach=True):
        if attach:
            tree.parents[pid] = parent
        att = Attempt(aid, parent, "Refine", program_id=pid, evaluations=[{"id": eid, "program_id": pid}])
        ledger.record_attempt(tree, archive, att)

    record(1, 1, 2, 2)
    record(2, 1, 3, 3)
    record(3, 2, 4, 4)
    record(4, 3, 5, 5, attach=False)
    record(5, 3, 6, 6, attach=False)

    options = [(None, ledger.direct[1]), (2, ledger.branch[2]), (3, ledger.branch[3])]
    _, rows = choose_direction(options, random.Random(0))
    by_target = {r["target"]: r for r in rows}

    # Latest steps have weight gamma^0 = 1, matching the exact worked example.
    assert by_target[None]["probability"] == pytest.approx(0.5)
    assert by_target[None]["estimate"] == pytest.approx(0.0625)
    assert by_target[2]["probability"] == pytest.approx(2 / 3)
    assert by_target[2]["estimate"] == pytest.approx(0.35 / 3)
    assert by_target[3]["probability"] == pytest.approx(0.25)
    assert by_target[3]["estimate"] == pytest.approx(0.0375)

    # Non-stationary recency decay: fresh breakthrough beats an ancient jump followed by stalls,
    # while a dead branch with 0 gains decays smoothly without saturating.
    ancient = History(8.5)
    ancient.observe(1, 8.1)  # gain 0.4 on step 1
    for k in range(2, 10):
        ancient.observe(k, 8.5)  # 8 consecutive stalls

    fresh = History(8.5)
    for k in range(1, 9):
        fresh.observe(k, 8.5)  # 8 stalls first
    fresh.observe(9, 8.1)  # fresh gain 0.4 on step 9

    stalled = History(7.9)
    for k in range(1, 21):
        stalled.observe(k, 7.9)  # 20 consecutive stalls

    _, cmp_rows = choose_direction(
        [("ancient", ancient), ("fresh", fresh), ("stalled", stalled)], random.Random(0)
    )
    by_name = {r["target"]: r for r in cmp_rows}
    assert by_name["ancient"]["s"] == pytest.approx(RECENCY_DECAY**8)
    assert by_name["fresh"]["s"] == pytest.approx(1.0)
    assert by_name["fresh"]["estimate"] > by_name["ancient"]["estimate"]
    assert by_name["stalled"]["probability"] == pytest.approx(1 / 22)

    # Path contraction transfers branch ledger seamlessly.
    ledger.reduce(Reduction("path", 1, 4, (2,)))
    assert 2 not in ledger.branch and 2 not in ledger.direct
    assert ledger.branch[4].count == 1 and ledger.branch[4].step_gains == [pytest.approx(0.4)]
    assert ledger.direct[1].count == 2

    parent, _ = choose_parent(
        TreePopulation(parents={1: ROOT_ID, 4: 1, 3: 1}), ledger, random.Random(1)
    )
    assert parent in {1, 3, 4}


def test_tree_lineage_donor_and_explore_reference_selection():
    # Tree:
    #   v0 -> 1 (7.5) -> 2 (7.2) -> 3 (7.0)
    #   v0 -> 4 (7.4) -> 5 (7.1)
    #                 -> 8 (7.8) -> 9 (7.7)
    #   v0 -> 6 (7.3) -> 7 (7.6)
    archive = {
        1: make_prog(1, 7.5),
        2: make_prog(2, 7.2),
        3: make_prog(3, 7.0),
        4: make_prog(4, 7.4),
        5: make_prog(5, 7.1),
        6: make_prog(6, 7.3),
        7: make_prog(7, 7.6),
        8: make_prog(8, 7.8),
        9: make_prog(9, 7.7),
    }
    tree = TreePopulation(
        parents={1: ROOT_ID, 2: 1, 3: 2, 4: ROOT_ID, 5: 4, 8: 4, 9: 8, 6: ROOT_ID, 7: 6}
    )

    assert tree_distance(tree, 2, 1) == 1
    assert tree_distance(tree, 2, 3) == 1
    assert tree_distance(tree, 2, 5) == 4  # 2->1->v0->4->5

    # Crossover for parent=2 must never pick its lineal parent (1) or child (3) when cross-branch nodes exist.
    others = [i for i in sorted(tree.parents) if i != 2]
    rng = random.Random(0)
    donors = {_select_donor(tree, others, 2, rng) for _ in range(50)}
    assert donors <= {4, 5, 6, 7, 8, 9}
    assert 1 not in donors and 3 not in donors

    # Explore for parent=2:
    # 1) First covers the best champion of each other top-level branch (5 from root 4, 6 from root 6);
    # 2) Then fills by max min-tree-distance from {2, 5, 6}: node 9 has min_dist=3 (to 5), beating adjacent nodes;
    # 3) Next slot ties at min_dist=1 and picks the best-scoring remaining node (3, fitness 7.0).
    refs = _select_explore_references(tree, archive, others, 2, limit=4)
    assert refs == [5, 6, 9, 3]

    # Single-chain fallback for Crossover still works cleanly.
    chain_tree = TreePopulation(parents={1: ROOT_ID, 2: 1, 3: 2})
    assert _select_donor(chain_tree, [1, 3], 2, random.Random(0)) in {1, 3}
    op, donor, erefs = choose_operator(tree, archive, 2, random.Random(2))
    assert op in {"Refine", "Crossover", "Explore"}
    if op == "Crossover":
        assert donor in {4, 5, 6, 7, 8, 9}
    elif op == "Explore":
        assert erefs[:3] == [5, 6, 9]


def test_exact_tie_and_no_positive_gains_fallback():
    h1, h2 = History(8.0), History(8.0)
    h1.observe(1, 8.0)
    _, rows = choose_direction([(1, h1), (2, h2)], random.Random(0))
    assert all(r["amplitude"] is None and r["estimate"] is None for r in rows)
    assert beats(make_prog(1, 7.9), make_prog(2, 8.0))
    assert not beats(make_prog(1, 8.0), make_prog(2, 8.0))
