from copy import deepcopy
from dataclasses import asdict
import json
import random

import pytest

from evotrace_v3.allocation import (
    AllocationConfig, allocate_step, eligible_reference_pool, lineage_peaks, parent_probabilities, select_parent,
)
from evotrace_v3.context import build_evolution_context
from evotrace_v3.models import Edge, Node, TreePopulation
from evotrace_v3.reflection import reflect_and_apply
from evotrace_v3.search import deserialize_tree
from evotrace_v3.topology import (
    PathContraction, SubtreePruning, TopologyConfig, apply_simplification, champion,
    plan_simplification, quality_scores,
)


def add(tree, parent, fit, *, op='Refine', reference=None, evaluations=1, code=None):
    i = tree.latest_node_id + 1
    tree.add(Node(i, code or f'def solve(x): return x + {i}', f'Idea {i}', fit,
                  failure='BAD_BRANCH' if fit is None else None),
             Edge(parent, i, op if parent else None, reference), evaluations=evaluations)
    return i


def test_population_ranks_ignore_units_and_respect_ties():
    tree = TreePopulation()
    for fit in [-20, -20, -10, 0]:
        add(tree, 0, fit)
    assert quality_scores(tree) == pytest.approx({1: 5/6, 2: 5/6, 3: 1/3, 4: 0})
    before = select_parent(tree, random.Random(4))
    for node in tree.nodes.values():
        node.fitness = 300 * node.fitness + 123
    assert select_parent(tree, random.Random(4)) == before
    for node in tree.nodes.values():
        node.fitness = 10
    assert set(quality_scores(tree).values()) == {.5}
    tree = TreePopulation()
    add(tree, 0, 10)
    assert quality_scores(tree) == {1: .5}


def test_new_best_competes_without_an_ancestor_gate_and_old_parent_remains_eligible():
    tree = TreePopulation()
    ancestor = add(tree, 0, 10)
    best = add(tree, ancestor, 1)
    other = add(tree, 0, 2)
    before = parent_probabilities(tree)
    # Past failures change only the ancestor's effort, not either candidate's quality.
    add(tree, ancestor, None, evaluations=100)
    after = parent_probabilities(tree)
    assert after[best] / after[other] == pytest.approx(before[best] / before[other])
    assert after[best] > after[other] > after[ancestor] > 0
    assert after[best] / after[ancestor] == pytest.approx(2 ** (2/10) * 102)


def test_quality_pressure_and_effort_balance_have_explicit_odds():
    tree = TreePopulation()
    for fit in range(21):
        add(tree, 0, fit)
    probabilities = parent_probabilities(tree)
    assert probabilities[1] / probabilities[11] == pytest.approx(2)
    assert probabilities[1] / probabilities[21] == pytest.approx(4)
    # One evaluation offsets a ten-position quality advantage, regardless of size.
    add(tree, 1, None)
    probabilities = parent_probabilities(tree)
    assert probabilities[1] == pytest.approx(probabilities[11])
    improved = add(tree, 1, -1)
    assert tree.nodes[improved].subtree_evaluations == 0
    probabilities = parent_probabilities(tree)
    assert probabilities[improved] > probabilities[1] > 0
    assert probabilities[21] > 0


def test_adding_worse_individuals_preserves_existing_pairwise_quality_odds():
    tree = TreePopulation()
    for fit in range(20):
        add(tree, 0, fit)
    before = parent_probabilities(tree)
    for fit in range(20, 160):
        add(tree, 0, fit)
    after = parent_probabilities(tree)
    assert after[1] / after[11] == pytest.approx(before[1] / before[11])
    assert after[1] < before[1]


def test_ties_singleton_empty_and_logged_sampling_probability():
    tree = TreePopulation()
    assert parent_probabilities(tree) == {}
    with pytest.raises(ValueError, match='no feasible'):
        select_parent(tree, random.Random(0))
    first = add(tree, 0, 1)
    assert parent_probabilities(tree) == {first: 1}
    second = add(tree, 0, 1)
    assert parent_probabilities(tree) == {first: .5, second: .5}
    class Capture(random.Random):
        def choices(self, population, weights, k):
            assert population == [first, second]
            assert weights == [.5, .5] and k == 1
            return [first]
    assert select_parent(tree, Capture()) == (first, .5)
    decision = allocate_step(tree, random.Random(2))
    assert decision.parent_probability == parent_probabilities(tree)[decision.parent_id]


@pytest.mark.parametrize('span', [1e-300, 1e300])
def test_extreme_rank_span_and_ties_stay_finite(span):
    tree = TreePopulation()
    for fit in [0, 0, 1]:
        add(tree, 0, fit)
    probabilities = parent_probabilities(tree, AllocationConfig(rank_half_life=span))
    assert sum(probabilities.values()) == pytest.approx(1)
    assert all(0 < probability <= 1 for probability in probabilities.values())
    assert probabilities[1] == probabilities[2]
    for node in tree.nodes.values():
        node.fitness = 1
    assert list(parent_probabilities(tree, AllocationConfig(rank_half_life=span)).values()) == [1/3]*3


def test_old_internal_individual_is_revisited_and_operator_labels_do_not_partition_tree():
    tree = TreePopulation()
    a = add(tree, 0, 10)
    b = add(tree, a, 1, op='Explore')
    c = add(tree, 0, 2)
    other = deepcopy(tree)
    other.edges[b].operator = 'Refine'
    assert select_parent(tree, random.Random(2)) == select_parent(other, random.Random(2))
    selected = set()
    for seed in range(60):
        parent, _ = select_parent(tree, random.Random(seed))
        selected.add(parent)
        add(tree, parent, None)
        tree = apply_simplification(tree, plan_simplification(tree))
    assert {a, b, c} <= selected


def test_actual_evaluations_include_explore_failures_and_repairs_while_reference_refreshes_last_used():
    tree = TreePopulation()
    a = add(tree, 0, 1)
    reference = add(tree, 0, 2)
    b = add(tree, a, 3, op='Explore', evaluations=2)
    assert tree.nodes[a].direct_evaluations == tree.nodes[a].subtree_evaluations == 2
    assert tree.nodes[a].last_used == 4
    assert tree.nodes[b].direct_evaluations == tree.nodes[b].subtree_evaluations == 0
    add(tree, b, None, op='Crossover', reference=reference, evaluations=2)
    assert tree.evaluations_used == 6
    assert tree.nodes[a].direct_evaluations == 2
    assert tree.nodes[a].subtree_evaluations == 4
    assert tree.nodes[b].direct_evaluations == tree.nodes[b].subtree_evaluations == 2
    assert tree.nodes[reference].direct_evaluations == tree.nodes[reference].subtree_evaluations == 0
    assert tree.nodes[reference].last_used == 6


def test_active_window_retains_unimproved_offspring_surpassed_ancestors_references_and_champion():
    tree = TreePopulation()
    champ = add(tree, 0, 1)
    ref_source = add(tree, 0, 5)
    unimproved = add(tree, champ, 2, op='Explore')
    surpassed_mid = add(tree, 0, 10)
    improved_tip = add(tree, surpassed_mid, 3)
    cfg_immediate = TopologyConfig(active_evaluations=40, region_size=1)
    plan = plan_simplification(tree, cfg_immediate)
    assert set(plan.kept_ids) == {champ, ref_source, unimproved, surpassed_mid, improved_tip}
    assert not plan.path_contractions and not plan.subtree_prunings

    tree.evaluations_used += 40
    cross_child = add(tree, champ, 4, op='Crossover', reference=ref_source)
    plan = plan_simplification(tree, cfg_immediate)
    assert set(plan.kept_ids) == {champ, ref_source, cross_child}
    assert tree.nodes[ref_source].direct_evaluations == 0


def test_region_size_waits_for_accumulated_attempts_and_simplifies_at_threshold():
    cfg = TopologyConfig(active_evaluations=10, region_size=4)
    tree = TreePopulation()
    fork = add(tree, 0, 10)
    m1 = add(tree, fork, 9)
    m2 = add(tree, m1, 8)
    branch_on_m1 = add(tree, m1, None)
    tip = add(tree, m2, 1)  # global champion
    other = add(tree, fork, 2)
    inactive_leaf = add(tree, other, 20)
    tree.evaluations_used += 10
    tree.nodes[other].last_used = tree.nodes[tip].last_used = tree.evaluations_used

    plan = plan_simplification(tree, cfg)
    assert not plan.path_contractions and not plan.subtree_prunings and not plan.triggers
    assert set(plan.kept_ids) == set(tree.nodes)
    assert inactive_leaf in parent_probabilities(tree)
    woke = deepcopy(tree)
    add(woke, inactive_leaf, None)
    assert inactive_leaf in plan_simplification(woke, TopologyConfig(active_evaluations=10, region_size=1)).kept_ids

    branch_on_m2 = add(tree, m2, None)
    tree.nodes[m2].last_used = 0
    plan = plan_simplification(tree, cfg)
    assert plan.path_contractions == (PathContraction(fork, tip, (m1, m2), (branch_on_m1, branch_on_m2)),)
    assert not plan.subtree_prunings
    assert inactive_leaf in plan.kept_ids
    assert plan.triggers == ({
        'kind': 'reflective_tree_simplification',
        'evaluations': tree.evaluations_used,
        'active_ids': sorted([tip, other]),
        'fork_ids': [fork],
        'removed_ids': sorted([m1, m2, branch_on_m1, branch_on_m2]),
    },)
    tree = apply_simplification(tree, plan)
    edge = tree.edges[tip]
    assert edge.parent_id == fork and edge.contracted and edge.operator is None and edge.reference_id is None

    f1 = add(tree, other, None)
    f2 = add(tree, other, None)
    assert not plan_simplification(tree, cfg).subtree_prunings
    f3 = add(tree, other, None)
    plan = plan_simplification(tree, cfg)
    assert plan.subtree_prunings == (SubtreePruning(other, (inactive_leaf, f1, f2, f3)),)


def test_experience_transfers_to_node_and_edge_without_inflating_descendant_effort():
    tree = TreePopulation()
    a = add(tree, 0, 10)
    b = add(tree, a, 9, op='Explore')
    c = add(tree, b, 8)
    d = add(tree, c, 1)
    branch = add(tree, b, None, evaluations=2)
    other = add(tree, a, 5)
    tree.nodes[b].experience = 'OLDER_NODE_EVIDENCE'
    tree.edges[c].experience = 'OLDER_EDGE_EVIDENCE'
    tree.evaluations_used += 40
    tree.nodes[other].last_used = tree.nodes[d].last_used = tree.evaluations_used
    before = deepcopy(tree)
    plan = plan_simplification(tree, TopologyConfig(active_evaluations=40, region_size=3))
    assert plan.kept_ids == (a, d, other)
    contraction, = plan.path_contractions
    assert contraction.removed_ids == (b, c) and contraction.branch_ids == (branch,)
    new, _ = reflect_and_apply(tree, plan, model=None)
    assert tree == before
    assert tree.nodes[b].subtree_evaluations == 4
    assert new.nodes[d].subtree_evaluations == tree.nodes[d].subtree_evaluations == 0
    assert new.nodes[d].direct_evaluations == 0
    assert new.nodes[a].direct_evaluations == 2
    for text in ['OLDER_NODE_EVIDENCE', 'OLDER_EDGE_EVIDENCE', 'BAD_BRANCH']:
        assert text in new.edges[d].experience
    assert new.nodes[a].experience == ''
    add(new, d, None, evaluations=2)
    assert new.nodes[d].direct_evaluations == 2
    assert new.nodes[d].subtree_evaluations == 2
    assert new.nodes[a].subtree_evaluations == 8


@pytest.mark.parametrize('cfg', [
    TopologyConfig(active_evaluations=4, region_size=2),
    TopologyConfig(active_evaluations=6, region_size=4),
])
@pytest.mark.parametrize('seed', range(8))
def test_budget_and_idempotence_survive_repeated_simplification_against_original_event_history(cfg, seed):
    rng = random.Random(seed)
    tree = TreePopulation()
    original_parents, events = {}, []
    saw_path = saw_subtree = False
    for _ in range(120):
        feasible = [i for i, node in tree.nodes.items() if node.fitness is not None]
        parent = rng.choice(feasible) if feasible else 0
        evaluations = rng.choice([1, 2])
        op = 'Refine'
        reference = None
        code = None
        if not parent:
            fit = rng.random()
        else:
            roll = rng.random()
            if roll < .15:
                fit = None
            elif roll < .25:
                donor = tree.nodes[rng.choice(feasible)]
                fit = donor.fitness
                code = f'# copy\n{donor.code}\n'
            elif roll < .35:
                fit = tree.nodes[parent].fitness
            else:
                fit = rng.random()
            other_feasible = [i for i in feasible if i != parent]
            if other_feasible and rng.random() < .25:
                op = 'Crossover'
                reference = rng.choice(other_feasible)
            else:
                op = rng.choice(['Refine', 'Explore'])
        i = add(tree, parent, fit, op=op, reference=reference, evaluations=evaluations, code=code)
        original_parents[i] = parent
        events.append((parent, evaluations))
        plan = plan_simplification(tree, cfg)
        saw_path |= bool(plan.path_contractions)
        saw_subtree |= bool(plan.subtree_prunings)
        for c in plan.path_contractions:
            assert len(c.removed_ids) + len(c.branch_ids) >= cfg.region_size
        for b in plan.subtree_prunings:
            assert len(b.removed_ids) >= cfg.region_size
        removed = [i for c in plan.path_contractions for i in (*c.removed_ids, *c.branch_ids)]
        removed += [i for b in plan.subtree_prunings for i in b.removed_ids]
        assert len(removed) == len(set(removed))
        assert set(removed) == set(tree.nodes) - set(plan.kept_ids)
        best = champion(tree)
        if best is not None:
            assert best in plan.kept_ids
        tree = apply_simplification(tree, plan)
        again = plan_simplification(tree, cfg)
        assert not again.path_contractions and not again.subtree_prunings and not again.triggers
        assert again.kept_ids == tuple(sorted(tree.nodes))
        assert tree.evaluations_used == sum(cost for _, cost in events)
        for i, node in tree.nodes.items():
            expected = 0
            assert node.direct_evaluations == sum(e for p, e in events if p == i)
            for actual_parent, cost in events:
                while actual_parent and actual_parent != i:
                    actual_parent = original_parents[actual_parent]
                if actual_parent == i:
                    expected += cost
            assert tree.nodes[i].subtree_evaluations == expected
            assert tree.edges[i].parent_id == 0 or tree.edges[i].parent_id in tree.nodes
        assert all(a in tree.nodes and b in tree.nodes for a, b in tree.used_crossover_pairs)
        assert deserialize_tree(json.loads(json.dumps(asdict(tree)))) == tree
    assert saw_path and saw_subtree


def test_reference_quality_uses_equal_ranks_for_ties_and_excludes_parent_first():
    tree = TreePopulation()
    for _ in range(4):
        add(tree, 0, 1)
    assert eligible_reference_pool(tree, 1) == [2, 3, 4]
    tree = TreePopulation()
    add(tree, 0, 1)
    add(tree, 0, 10)
    assert eligible_reference_pool(tree, 1) == [2]


@pytest.mark.parametrize('span', [0, -1, float('nan'), float('inf')])
def test_rank_span_is_finite_and_positive(span):
    with pytest.raises(ValueError):
        AllocationConfig(rank_half_life=span)


def test_three_round_reflection_reuses_old_notes_and_reaches_all_operators():
    t = TreePopulation()
    a = add(t, 0, 10)
    other = add(t, a, 5)
    tip = a
    prompts = []
    class Notes:
        def complete(self, prompt, **kwargs):
            prompts.append(prompt)
            marker = 'EDGE' if '[Path Evidence]' in prompt else 'NODE'
            return json.dumps({'experience': f'{marker}_NOTE_{round_number}'})
    for round_number in range(3):
        mid = add(t, tip, 9)
        tip = add(t, mid, 1/(round_number+1))
        add(t, mid, None)
        add(t, a, None)
        t.evaluations_used += 40
        t.nodes[other].last_used = t.nodes[tip].last_used = t.evaluations_used
        plan = plan_simplification(t, TopologyConfig(active_evaluations=40, region_size=1))
        assert plan.path_contractions and plan.subtree_prunings
        prompt_start = len(prompts)
        t, _ = reflect_and_apply(t, plan, model=Notes())
        if round_number:
            used = '\n'.join(prompts[prompt_start:])
            assert f'EDGE_NOTE_{round_number-1}' in used
            assert f'NODE_NOTE_{round_number-1}' in used
        for op, reference in [('Refine', None), ('Explore', None), ('Crossover', a)]:
            context, _, _, _ = build_evolution_context(t, tip, op, reference)
            assert f'EDGE_NOTE_{round_number}' in context
            assert f'NODE_NOTE_{round_number}' in context
        t = deserialize_tree(json.loads(json.dumps(asdict(t))))


def test_no_population_cap_and_no_structural_distance(monkeypatch):
    from evotrace_v3 import topology
    def no_distance(*args):
        pytest.fail('Structural distance must not decide survival')
    monkeypatch.setattr(topology, 'code_distance', no_distance)
    tree = TreePopulation()
    lines = [add(tree, 0, 100 + i) for i in range(5)]
    for step in range(60):
        lines.append(add(tree, lines[step % 5], 99 - step))
    plan = plan_simplification(tree, TopologyConfig(active_evaluations=100, region_size=4))
    assert not plan.path_contractions and not plan.subtree_prunings
    assert len(plan.kept_ids) == 65


@pytest.mark.parametrize('operator,copy_parent', [('Refine', 'original'), ('Crossover', 'other')])
def test_duplicate_program_does_not_create_a_fresh_sampling_weight(operator, copy_parent, monkeypatch):
    from evotrace_v3 import topology
    def no_distance(*args):
        pytest.fail('Structural distance must not decide survival')
    monkeypatch.setattr(topology, 'code_distance', no_distance)
    tree = TreePopulation()
    original = add(tree, 0, 1, code='def solve(x): return x + 1')
    other = add(tree, 0, 2, code='def solve(x): return x * 2')
    add(tree, original, None, evaluations=7)
    parent = original if copy_parent == 'original' else other
    copy_id = add(tree, parent, 1, op=operator,
                  reference=original if operator == 'Crossover' else None,
                  code='# same source\ndef solve(x):\n    return x + 1\n')
    assert set(parent_probabilities(tree)) == {original, other}
    assert copy_id not in eligible_reference_pool(tree, other)
    pending_plan = plan_simplification(tree)
    assert not pending_plan.subtree_prunings and copy_id in pending_plan.kept_ids

    before = deepcopy(tree)
    plan = plan_simplification(tree, TopologyConfig(active_evaluations=40, region_size=1))
    assert copy_id not in plan.kept_ids
    new, _ = reflect_and_apply(tree, plan, model=None)
    assert set(parent_probabilities(new)) == {original, other}
    assert new.evaluations_used == before.evaluations_used
    for i in (original, other):
        assert new.nodes[i].subtree_evaluations == before.nodes[i].subtree_evaluations
        assert new.nodes[i].last_used == before.nodes[i].last_used
    assert new.nodes[parent].experience
    changed = add(new, parent, .5, code='def solve(x): return x + 1')
    assert changed in parent_probabilities(new)
    assert changed in plan_simplification(new, TopologyConfig(active_evaluations=40, region_size=1)).kept_ids


def test_lca_closure_retains_forks_and_simplification_is_idempotent():
    tree = TreePopulation()
    fork = add(tree, 0, 100)
    m1 = add(tree, fork, 90)
    m2 = add(tree, m1, 80)
    m3 = add(tree, m2, 70)
    left = add(tree, m3, 1)
    right = add(tree, fork, 2)
    discarded = add(tree, m2, 200)
    tree.evaluations_used += 40
    tree.nodes[right].last_used = tree.evaluations_used
    plan = plan_simplification(tree)
    assert set(plan.kept_ids) == {fork, left, right}
    assert plan.triggers[0]['kind'] == 'reflective_tree_simplification'
    assert plan.triggers[0]['fork_ids'] == [fork]
    assert plan.triggers[0]['removed_ids'] == [m1, m2, m3, discarded]
    contraction, = plan.path_contractions
    assert (contraction.parent_id, contraction.child_id, contraction.removed_ids, contraction.branch_ids) == (
        fork, left, (m1, m2, m3), (discarded,))
    new = apply_simplification(tree, plan)
    again = plan_simplification(new)
    assert again.kept_ids == plan.kept_ids
    assert not again.path_contractions and not again.subtree_prunings
    assert set(parent_probabilities(new)) == {fork, left, right}


@pytest.mark.parametrize('field', ['active_evaluations', 'region_size'])
@pytest.mark.parametrize('value', [0, -1, 1.5, True])
def test_trigger_parameters_are_positive_integer_counts(field, value):
    with pytest.raises(ValueError, match=field):
        TopologyConfig(**{field: value})


def test_regressed_and_tied_descendants_share_lineage_peak_and_never_invert_parent():
    tree = TreePopulation()
    peak = add(tree, 0, 5.70)
    step1 = add(tree, peak, 5.72)
    for _ in range(5):
        add(tree, step1, None)
    step2 = add(tree, step1, 5.73)
    rebound = add(tree, step2, 5.71)
    tied_rewrite = add(tree, peak, 5.70, code='def solve(x): return x * 1 + 1')
    peaks = lineage_peaks(tree)
    for nid in (peak, step1, step2, rebound, tied_rewrite):
        assert peaks[nid] == peak
    probs = parent_probabilities(tree)
    assert probs[peak] == pytest.approx(probs[tied_rewrite])
    assert probs[peak] > probs[rebound] > probs[step1] > probs[step2] > 0


def test_non_improving_family_total_weight_is_bounded_by_peak_base_weight():
    cfg = AllocationConfig(rank_half_life=10.0)
    for k in (1, 5, 25):
        tree = TreePopulation()
        peak = add(tree, 0, 10.0)
        rival = add(tree, 0, 10.5)
        family = [peak]
        for idx in range(k):
            family.append(add(tree, peak, 10.01 + 0.001 * idx))
        probs = parent_probabilities(tree, cfg)
        quality = quality_scores(tree)
        best_q = max(quality.values())
        n = len(quality)
        peak_rank = (best_q - quality[peak]) * (n - 1)
        rival_rank = (best_q - quality[rival]) * (n - 1)
        family_sum = sum(probs[nid] for nid in family)
        max_ratio = 2.0 ** ((rival_rank - peak_rank) / cfg.rank_half_life)
        assert family_sum / probs[rival] <= max_ratio + 1e-12


def test_deep_breakthrough_resets_lineage_effort_and_lighter_branch_beats_heavier_branch():
    tree = TreePopulation()
    heavy = add(tree, 0, 2.0)
    light = add(tree, 0, 2.0, code='def solve(x): return 2 * 1')
    for _ in range(20):
        add(tree, heavy, None)
    for _ in range(2):
        add(tree, light, None)
    probs = parent_probabilities(tree)
    assert probs[light] / probs[heavy] == pytest.approx((1 + 20) / (1 + 2))

    cur = heavy
    for depth in range(1, 8):
        cur = add(tree, cur, 2.0 + depth)
    breakthrough = add(tree, cur, 1.0)
    assert lineage_peaks(tree)[breakthrough] == breakthrough
    assert tree.nodes[breakthrough].subtree_evaluations == 0
    probs_after = parent_probabilities(tree)
    assert probs_after[breakthrough] > probs_after[light] > probs_after[heavy]
