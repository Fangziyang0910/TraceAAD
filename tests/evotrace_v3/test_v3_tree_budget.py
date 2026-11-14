from copy import deepcopy
from dataclasses import asdict
import json
import random

import pytest

from evotrace_v3.allocation import (
    AllocationConfig, allocate_step, eligible_reference_pool, parent_probabilities, select_parent,
)
from evotrace_v3.context import build_evolution_context
from evotrace_v3.models import Edge, Node, TreePopulation
from evotrace_v3.reflection import reflect_and_apply
from evotrace_v3.search import deserialize_tree
from evotrace_v3.topology import (
    TopologyConfig, apply_consolidation, plan_consolidation, quality_representatives, quality_scores,
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
    assert after[best] / after[ancestor] == pytest.approx(10 * 102)


def test_quality_pressure_and_effort_balance_have_explicit_odds():
    tree = TreePopulation()
    best = add(tree, 0, 1)
    weak = add(tree, 0, 10)
    probabilities = parent_probabilities(tree)
    assert probabilities[best] / probabilities[weak] == pytest.approx(10)
    # Nine actual evaluations offset the default tenfold quality advantage.
    add(tree, best, None, evaluations=9)
    assert parent_probabilities(tree) == pytest.approx({best: .5, weak: .5})
    # A new improvement retains its own zero effort; no ancestry debt is inherited.
    improved = add(tree, best, 0)
    assert tree.nodes[improved].subtree_evaluations == 0
    probabilities = parent_probabilities(tree)
    assert probabilities[improved] > probabilities[best] > 0
    assert probabilities[weak] > 0


def test_ties_singleton_empty_and_logged_sampling_probability():
    tree = TreePopulation()
    assert parent_probabilities(tree) == {}
    with pytest.raises(ValueError, match='no feasible'):
        select_parent(tree, random.Random(0))
    first = add(tree, 0, 1)
    assert parent_probabilities(tree) == {first: 1}
    second = add(tree, 0, 1)
    assert parent_probabilities(tree) == {first: .5, second: .5}
    # Verify the sampler uses the whole distribution, rather than argmax.
    class Capture(random.Random):
        def choices(self, population, weights, k):
            assert population == [first, second]
            assert weights == [.5, .5] and k == 1
            return [first]
    assert select_parent(tree, Capture()) == (first, .5)
    decision = allocate_step(tree, random.Random(2))
    assert decision.parent_probability == parent_probabilities(tree)[decision.parent_id]


def test_quality_ratio_one_removes_quality_preference_and_extreme_ratio_stays_finite():
    tree = TreePopulation()
    for fit in range(3):
        add(tree, 0, fit)
    assert list(parent_probabilities(tree, AllocationConfig(quality_ratio=1)).values()) == [1/3]*3
    probabilities = parent_probabilities(tree, AllocationConfig(quality_ratio=1e300))
    assert sum(probabilities.values()) == pytest.approx(1)
    assert all(0 < probability <= 1 for probability in probabilities.values())


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
        tree = apply_consolidation(tree, plan_consolidation(
            tree, TopologyConfig(quality_floor=0, clone_distance=0)))
    assert {a, b, c} <= selected


def test_actual_evaluations_include_explore_failures_and_repairs_but_not_reference_reads():
    tree = TreePopulation()
    a = add(tree, 0, 1)
    reference = add(tree, 0, 2)
    b = add(tree, a, 3, op='Explore', evaluations=2)
    assert tree.nodes[a].development_evaluations == tree.nodes[a].subtree_evaluations == 2
    assert tree.nodes[a].last_used == 4
    assert tree.nodes[b].development_evaluations == tree.nodes[b].subtree_evaluations == 0
    add(tree, b, None, op='Crossover', reference=reference, evaluations=2)
    assert tree.evaluations_used == 6
    assert tree.nodes[a].development_evaluations == 2
    assert tree.nodes[a].subtree_evaluations == 4
    assert tree.nodes[b].development_evaluations == tree.nodes[b].subtree_evaluations == 2
    assert tree.nodes[reference].development_evaluations == tree.nodes[reference].subtree_evaluations == 0
    assert tree.nodes[reference].last_used == 2


def test_retention_uses_individual_quality_difference_and_own_activity():
    tree = TreePopulation()
    a = add(tree, 0, 5, code='def solve(x): return sum(x)')
    add(tree, a, 1, code='def solve(x): return x[0]')
    child = add(tree, a, 9)
    add(tree, 0, 10)
    tree.evaluations_used += 40
    tree.nodes[a].last_used = tree.evaluations_used
    cfg = TopologyConfig(clone_distance=0)
    assert a in quality_representatives(tree, cfg)  # A better child does not erase a distinct parent.
    assert child not in plan_consolidation(tree, cfg).kept_ids  # Parent activity does not refresh siblings.


def test_experience_moves_but_descendant_does_not_inherit_ancestor_effort():
    tree = TreePopulation()
    a = add(tree, 0, 10)
    b = add(tree, a, 9, op='Explore')
    c = add(tree, b, 8)
    d = add(tree, c, 1)
    side = add(tree, b, None, evaluations=2)
    other = add(tree, a, 5)
    tree.nodes[b].selection_experience = 'OLDER_BRANCH_EVIDENCE'
    tree.edges[c].variation_experience = 'OLDER_PATH_EVIDENCE'
    tree.evaluations_used += 40
    tree.nodes[other].last_used = tree.nodes[d].last_used = tree.evaluations_used
    before = deepcopy(tree)
    plan = plan_consolidation(tree, TopologyConfig(quality_floor=1))
    assert plan.kept_ids == (a, d, other)
    contraction, = plan.path_contractions
    assert contraction.removed_ids == (b, c) and contraction.side_ids == (side,)
    new, _ = reflect_and_apply(tree, plan, model=None)
    assert tree == before
    assert tree.nodes[b].subtree_evaluations == 4
    assert new.nodes[d].subtree_evaluations == tree.nodes[d].subtree_evaluations == 0
    assert new.nodes[d].development_evaluations == 0
    assert new.nodes[a].development_evaluations == 2
    for text in ['OLDER_BRANCH_EVIDENCE', 'OLDER_PATH_EVIDENCE', 'BAD_BRANCH']:
        assert text in new.edges[d].variation_experience
    assert new.nodes[a].selection_experience == ''
    add(new, d, None, evaluations=2)
    assert new.nodes[d].development_evaluations == 2
    assert new.nodes[d].subtree_evaluations == 2
    assert new.nodes[a].subtree_evaluations == 8


@pytest.mark.parametrize('seed', range(5))
def test_budget_survives_repeated_simplification_against_original_event_history(seed):
    rng = random.Random(seed)
    tree = TreePopulation()
    original_parents, events = {}, []
    saw_path = saw_side = False
    for _ in range(100):
        parent = rng.choice(list(tree.nodes)) if tree.nodes else 0
        evaluations = rng.choice([1, 2])
        fit = None if parent and rng.random() < .15 else rng.random()
        i = add(tree, parent, fit, op=rng.choice(['Refine', 'Explore']), evaluations=evaluations)
        original_parents[i] = parent
        events.append((parent, evaluations))
        plan = plan_consolidation(tree, TopologyConfig(cold_evaluations=4))
        saw_path |= bool(plan.path_contractions)
        saw_side |= bool(plan.branch_consolidations)
        removed = [i for c in plan.path_contractions for i in (*c.removed_ids, *c.side_ids)]
        removed += [i for b in plan.branch_consolidations for i in b.removed_ids]
        assert len(removed) == len(set(removed))
        assert set(removed) == set(tree.nodes) - set(plan.kept_ids)
        tree = apply_consolidation(tree, plan)
        assert tree.evaluations_used == sum(cost for _, cost in events)
        for i, node in tree.nodes.items():
            expected = 0
            assert node.development_evaluations == sum(e for p, e in events if p == i)
            for actual_parent, cost in events:
                while actual_parent and actual_parent != i:
                    actual_parent = original_parents[actual_parent]
                if actual_parent == i:
                    expected += cost
            assert tree.nodes[i].subtree_evaluations == expected
            assert tree.edges[i].parent_id == 0 or tree.edges[i].parent_id in tree.nodes
        assert deserialize_tree(json.loads(json.dumps(asdict(tree)))) == tree
    assert saw_path and saw_side


def test_reference_quality_uses_equal_ranks_for_ties_and_excludes_parent_first():
    tree = TreePopulation()
    for _ in range(4):
        add(tree, 0, 1)
    assert eligible_reference_pool(tree, 1) == [2, 3, 4]
    tree = TreePopulation()
    add(tree, 0, 1)
    add(tree, 0, 10)
    assert eligible_reference_pool(tree, 1) == [2]


@pytest.mark.parametrize('ratio', [0, .5, -1, float('nan'), float('inf')])
def test_quality_ratio_is_finite_and_at_least_one(ratio):
    with pytest.raises(ValueError):
        AllocationConfig(quality_ratio=ratio)


def test_three_round_reflection_reuses_old_notes_and_reaches_all_operators():
    t = TreePopulation()
    a = add(t, 0, 10)
    other = add(t, a, 5)  # a second live line keeps a as a fork
    tip = a
    prompts = []
    class Notes:
        def complete(self, prompt, **kwargs):
            prompts.append(prompt)
            marker = 'PATH' if '[Path Evidence]' in prompt else 'SIDE'
            return json.dumps({'experience': f'{marker}_NOTE_{round_number}'})
    for round_number in range(3):
        mid = add(t, tip, 9)
        tip = add(t, mid, 1/(round_number+1))
        add(t, mid, None)  # side off contracted `mid` joins the reconnected edge
        add(t, a, None)    # side off retained `a` joins `a.selection_experience`
        t.evaluations_used += 40
        t.nodes[other].last_used = t.nodes[tip].last_used = t.evaluations_used
        plan = plan_consolidation(t, TopologyConfig(quality_floor=1))
        assert plan.path_contractions and plan.branch_consolidations
        prompt_start = len(prompts)
        t, _ = reflect_and_apply(t, plan, model=Notes())
        if round_number:
            used = '\n'.join(prompts[prompt_start:])
            assert f'PATH_NOTE_{round_number-1}' in used
            assert f'SIDE_NOTE_{round_number-1}' in used
        for op, reference in [('Refine', None), ('Explore', None), ('Crossover', a)]:
            context, _, _, _ = build_evolution_context(t, tip, op, reference)
            assert f'PATH_NOTE_{round_number}' in context
            assert f'SIDE_NOTE_{round_number}' in context
        t = deserialize_tree(json.loads(json.dumps(asdict(t))))

