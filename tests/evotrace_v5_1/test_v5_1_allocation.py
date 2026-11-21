"""竞争分配与三算子选材测试：中位数/MAD Sigmoid 静音老分叉、投入折扣、跨谱系供体与探索代表。"""

import random

from evotrace_v5_1.allocation import (
    choose_donor,
    choose_step,
    explore_leaders,
    log_quality_weights,
    parent_probabilities,
)
from evotrace_v5_1.models import ROOT_ID, Evaluation, Program, TreePopulation


def make_prog(pid: int, fitness: float, *, parent_id: int = ROOT_ID) -> Program:
    return Program(
        pid,
        f"def solve(x):\n    return {pid}\n",
        f"Design {pid}",
        "Init",
        parent_id,
        None,
        Evaluation("success", fitness),
    )


def test_median_mad_sigmoid_silences_weak_fork_anchors_and_applies_effort_discount():
    # Elite nodes near 10.0, weak fork anchor at 50.0 (far worse than median)
    archive = {
        1: make_prog(1, 50.0),  # old weak fork anchor
        2: make_prog(2, 10.0, parent_id=1),
        3: make_prog(3, 10.1, parent_id=1),
        4: make_prog(4, 10.2),
        5: make_prog(5, 10.5),
        6: make_prog(6, 11.0),
    }
    tree = TreePopulation(
        parents={1: ROOT_ID, 2: 1, 3: 1, 4: ROOT_ID, 5: ROOT_ID, 6: ROOT_ID},
        direct_evals={1: 0, 2: 3, 3: 0, 4: 0, 5: 0, 6: 0},
    )
    probs, _, meta = parent_probabilities(tree, archive, selection_lambda=10.0)
    assert abs(sum(probs.values()) - 1.0) < 1e-12
    assert meta["population_size"] == 6 and meta["active_lineages"] == 4
    # Weak fork anchor must be near-silenced automatically without any hard exclusion ban
    assert probs[1] < 1e-6
    # Fresh strong node 3 (n=0) gets higher probability than heavily exploited node 2 (n=3)
    assert probs[3] > probs[2]


def test_sigmoid_handles_all_tied_scores_and_extreme_outliers_without_underflow():
    archive = {i: make_prog(i, 10.0) for i in range(1, 6)}
    tree = TreePopulation(parents={i: ROOT_ID for i in range(1, 6)})
    probs, _, _ = parent_probabilities(tree, archive)
    assert all(abs(p - 0.2) < 1e-9 for p in probs.values())

    # Extreme outlier among tied scores: MAD=0 falls back to outlier scale, still crushing it to ~1.8e-5
    archive[6] = make_prog(6, 1e9)
    tree.parents[6] = ROOT_ID
    probs2, log_q, _ = parent_probabilities(tree, archive)
    assert abs(sum(probs2.values()) - 1.0) < 1e-12
    assert probs2[6] < 1e-4
    assert all(v <= 0.0 for v in log_q.values())

    # Extreme outlier with normal non-zero MAD is suppressed to essentially zero without underflow
    archive[2] = make_prog(2, 10.5)
    archive[3] = make_prog(3, 11.0)
    probs3, _, _ = parent_probabilities(tree, archive)
    assert abs(sum(probs3.values()) - 1.0) < 1e-12
    assert probs3[6] < 1e-12


def test_cross_lineage_donor_prefers_other_lineages_and_falls_back_cleanly():
    # Lineage 1: {1, 2}, Lineage 3: {3, 4}
    archive = {
        1: make_prog(1, 5.0),
        2: make_prog(2, 6.0, parent_id=1),
        3: make_prog(3, 12.0),
        4: make_prog(4, 15.0, parent_id=3),
    }
    tree = TreePopulation(parents={1: ROOT_ID, 2: 1, 3: ROOT_ID, 4: 3})
    log_q, _, _ = log_quality_weights(tree, archive, selection_lambda=10.0)
    rng = random.Random(7)
    # Even though lineage 3 is much worse than median, cross-lineage donor must come from {3, 4}, favoring 3
    donors = [choose_donor(tree, log_q, 1, rng) for _ in range(100)]
    assert all(is_cross and d in {3, 4} for d, is_cross in donors)
    assert sum(d == 3 for d, _ in donors) > 90

    # Single lineage falls back to V \ {p}
    tree_single = TreePopulation(parents={1: ROOT_ID, 2: 1})
    log_q_s, _, _ = log_quality_weights(tree_single, archive)
    donor, is_cross = choose_donor(tree_single, log_q_s, 1, rng)
    assert donor == 2 and is_cross is False

    # Single node falls back to None (step switches Crossover -> Refine)
    tree_one = TreePopulation(parents={1: ROOT_ID})
    donor1, _ = choose_donor(tree_one, {1: 0.0}, 1, rng)
    assert donor1 is None


def test_explore_leaders_never_pads_with_same_lineage_nodes_and_sorts_best_first():
    # Only 2 active top-level lineages even though 6 nodes exist and limit=5
    archive = {
        1: make_prog(1, 20.0),
        2: make_prog(2, 8.0, parent_id=1),
        3: make_prog(3, 9.0, parent_id=1),
        4: make_prog(4, 15.0),
        5: make_prog(5, 6.0, parent_id=4),
        6: make_prog(6, 7.0, parent_id=4),
    }
    tree = TreePopulation(parents={1: ROOT_ID, 2: 1, 3: 1, 4: ROOT_ID, 5: 4, 6: 4})
    leaders = explore_leaders(tree, archive, limit=5)
    # Must contain strictly 2 leaders (one per distinct lineage), sorted best-to-worst: (5, 2)
    assert leaders == (5, 2)


def test_operator_priors_match_045_030_025_and_explore_attaches_to_root():
    archive = {i: make_prog(i, float(i)) for i in range(1, 7)}
    tree = TreePopulation(parents={i: ROOT_ID for i in range(1, 7)})
    rng = random.Random(123)
    counts = {"Refine": 0, "Explore": 0, "Crossover": 0}
    for _ in range(2000):
        parent, op, ref, leaders, trace = choose_step(tree, archive, rng, explore_references=5)
        counts[op] += 1
        if op == "Explore":
            assert parent == ROOT_ID and ref is None and len(leaders) == 5
        elif op == "Crossover":
            assert parent != ROOT_ID and ref is not None and ref != parent
            assert trace["cross_lineage"] is True
        else:
            assert parent != ROOT_ID and ref is None and leaders == ()
    assert 0.41 <= counts["Refine"] / 2000 <= 0.49
    assert 0.26 <= counts["Explore"] / 2000 <= 0.34
    assert 0.21 <= counts["Crossover"] / 2000 <= 0.29
