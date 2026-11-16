from copy import deepcopy
from dataclasses import asdict
import json
import random

import pytest

from evotrace_v3.allocation import (
    AllocationConfig,
    allocate_step,
    eligible_reference_pool,
    parent_probabilities,
    select_parent,
)
from evotrace_v3.context import build_evolution_context
from evotrace_v3.models import Edge, Node, TreePopulation
from evotrace_v3.reflection import reflect_and_apply
from evotrace_v3.search import deserialize_tree
from evotrace_v3.topology import (
    PathContraction,
    SubtreePruning,
    TopologyConfig,
    apply_simplification,
    champion,
    code_distance,
    plan_simplification,
    quality_scores,
    representatives,
)


DISTINCT_BODIES = [
    "return x + 1",
    "return x * x - 1",
    "return sum(i for i in range(x))",
    "return max(x, -x) // 2",
    "return min(i * 3 for i in range(1, x + 2))",
    "return len([i for i in range(x) if i % 2])",
    "return abs(x - 7) ** 2",
    "return int(pow(x + 2, 0.5))",
    "return sum(i ^ 3 for i in range(x))",
    "return all(i >= 0 for i in range(x)) and x",
    "return any(i == x for i in range(5)) or x",
    "return sorted([x, x - 1, x + 2])[1]",
    "return tuple(range(x)).count(1)",
    "return dict(enumerate(range(x))).get(0, x)",
    "return {i % 5 for i in range(x)}.pop() if x > 0 else 0",
    "return (x << 2) & (x | 7)",
    "return round(float(x) / 3.0, 2)",
    "return sum(divmod(abs(x) + 3, 2))",
    "return next((i for i in range(x) if i > 2), x)",
    "return list(reversed(range(x + 1)))[0]",
    "return sum(1 for _ in zip(range(x), range(x)))",
    "return (x % 7) * (x // 3) - (x % 2)",
    "return bool(x) + (x ^ 15)",
    "return sum(map(abs, [x, -x, x - 1]))",
    "return min([x + 4, x * 2, x ^ 1])",
]


def distinct_code(index: int) -> str:
    body = DISTINCT_BODIES[index % len(DISTINCT_BODIES)]
    extra = "\n".join(f"    v_{k} = (x + {k}) % ({k + 2})" for k in range(index // len(DISTINCT_BODIES)))
    prefix = (extra + "\n") if extra else ""
    return f"def solve(x):\n{prefix}    {body}\n"


def add(tree, parent, fit, *, op="Refine", reference=None, evaluations=1, code=None):
    i = tree.latest_node_id + 1
    src = code if code is not None else distinct_code(i - 1)
    tree.add(
        Node(
            i,
            src,
            f"Idea {i}",
            fit,
            failure="BAD_BRANCH" if fit is None else None,
        ),
        Edge(parent, i, op if parent else None, reference),
        evaluations=evaluations,
    )
    return i


def test_population_ranks_ignore_units_and_respect_ties():
    tree = TreePopulation()
    for fit in [-20, -20, -10, 0]:
        add(tree, 0, fit)
    assert quality_scores(tree) == pytest.approx({1: 5 / 6, 2: 5 / 6, 3: 1 / 3, 4: 0})
    top = TopologyConfig(max_representatives=24, cover_distance=0.0)
    before = select_parent(tree, random.Random(4), topology_config=top)
    for node in tree.nodes.values():
        node.fitness = 300 * node.fitness + 123
    assert select_parent(tree, random.Random(4), topology_config=top) == before
    for node in tree.nodes.values():
        node.fitness = 10
    assert set(quality_scores(tree).values()) == {0.5}
    tree = TreePopulation()
    add(tree, 0, 10)
    assert quality_scores(tree) == {1: 0.5}


def test_quality_pressure_and_direct_effort_have_explicit_odds():
    top = TopologyConfig(max_representatives=32, cover_distance=0.0)
    tree = TreePopulation()
    for fit in range(21):
        add(tree, 0, fit)
    probabilities = parent_probabilities(tree, topology_config=top)
    assert probabilities[1] / probabilities[11] == pytest.approx(2)
    assert probabilities[1] / probabilities[21] == pytest.approx(4)
    # One direct evaluation halves parent 1's weight, matching a 10-rank gap.
    add(tree, 1, None)
    probabilities = parent_probabilities(tree, topology_config=top)
    assert probabilities[1] == pytest.approx(probabilities[11])
    improved = add(tree, 1, -1)
    assert tree.nodes[improved].direct_evaluations == 0
    probabilities = parent_probabilities(tree, topology_config=top)
    assert probabilities[improved] > probabilities[1] > 0
    assert probabilities[21] > 0


def test_ties_singleton_empty_and_logged_sampling_probability():
    top = TopologyConfig(max_representatives=24, cover_distance=0.15)
    tree = TreePopulation()
    assert parent_probabilities(tree, topology_config=top) == {}
    with pytest.raises(ValueError, match="no feasible"):
        select_parent(tree, random.Random(0), topology_config=top)
    first = add(tree, 0, 1, code="def solve(x):\n    return x + 1\n")
    assert parent_probabilities(tree, topology_config=top) == {first: 1.0}
    second = add(tree, 0, 1, code="def solve(x):\n    return sum(i * i for i in range(x))\n")
    assert code_distance(tree.nodes[first].code, tree.nodes[second].code) >= 0.15
    assert parent_probabilities(tree, topology_config=top) == {first: 0.5, second: 0.5}

    class Capture(random.Random):
        def choices(self, population, weights, k):
            assert population == [first, second]
            assert weights == [0.5, 0.5] and k == 1
            return [first]

    assert select_parent(tree, Capture(), topology_config=top) == (first, 0.5)
    decision = allocate_step(tree, random.Random(2), topology_config=top)
    assert decision.parent_probability == parent_probabilities(tree, topology_config=top)[decision.parent_id]


@pytest.mark.parametrize("span", [1e-300, 1e300])
def test_extreme_rank_span_and_ties_stay_finite(span):
    top = TopologyConfig(max_representatives=24, cover_distance=0.0)
    tree = TreePopulation()
    for fit in [0, 0, 1]:
        add(tree, 0, fit)
    probabilities = parent_probabilities(tree, AllocationConfig(rank_half_life=span), top)
    assert sum(probabilities.values()) == pytest.approx(1)
    assert all(0 < probability <= 1 for probability in probabilities.values())
    assert probabilities[1] == probabilities[2]
    for node in tree.nodes.values():
        node.fitness = 1
    assert list(parent_probabilities(tree, AllocationConfig(rank_half_life=span), top).values()) == [1 / 3] * 3


def test_representative_cover_collapses_clones_supersedes_constants_and_frees_explore_from_ancestor_effort():
    cfg = TopologyConfig(max_representatives=4, cover_distance=0.15, region_size=2)
    tree = TreePopulation()
    # Heavy champion with 25 failed/worse evaluations
    champ = add(tree, 0, 8.30, code="def solve(x):\n    return x + 10\n")
    for k in range(25):
        # Worse constant/layout clones (distance 0.0 < 0.15)
        add(tree, champ, 8.31 + 0.01 * k, code=f"def solve(x):\n    return x + {20 + k}\n")
    rival = add(tree, 0, 8.40, code="def solve(x):\n    return sum(i * i for i in range(x))\n")

    # None of the 25 worse clones enter R or crowd `rival`'s rank!
    assert representatives(tree, cfg) == (champ, rival)
    probs = parent_probabilities(tree, topology_config=cfg)
    assert set(probs) == {champ, rival}
    # Champ has direct_evaluations=25, rival has 0 -> rival gets majority of budget!
    assert probs[rival] > probs[champ]

    # A structurally distinct Explore child under heavy `champ` starts with direct_evaluations == 0!
    novel = add(
        tree,
        champ,
        8.35,
        op="Explore",
        code="def solve(x):\n    acc = 1\n    for idx in range(1, x + 1):\n        acc = (acc * 31) ^ idx\n    return acc\n",
    )
    assert code_distance(tree.nodes[champ].code, tree.nodes[novel].code) >= 0.15
    assert representatives(tree, cfg) == (champ, novel, rival)
    assert tree.edges[novel].parent_id == champ  # Real tree ancestry preserved!
    assert tree.nodes[novel].direct_evaluations == 0
    probs_after_explore = parent_probabilities(tree, topology_config=cfg)
    assert probs_after_explore[novel] > probs_after_explore[champ]

    # A strictly better constant tweak (`d == 0.0 < 0.15`) supersedes `champ` in R!
    better_const = add(tree, champ, 8.25, op="Refine", code="def solve(x):\n    return x + 5\n")
    assert code_distance(tree.nodes[champ].code, tree.nodes[better_const].code) == 0.0
    assert representatives(tree, cfg) == (better_const, novel, rival)
    assert champ not in parent_probabilities(tree, topology_config=cfg)
    # `champ` is still retained as LCA fork connecting `better_const` and `novel`, WITHOUT blocking `better_const`!
    plan = plan_simplification(tree, cfg)
    assert champ in plan.kept_ids
    assert champ in plan.triggers[0]["fork_ids"]
    assert champ not in plan.triggers[0]["representative_ids"]


def test_duplicate_program_conserves_existing_direct_effort_and_never_creates_fresh_weight():
    cfg = TopologyConfig(max_representatives=8, cover_distance=0.15, region_size=1)
    tree = TreePopulation()
    original = add(tree, 0, 1.0, code="def solve(x):\n    return x + 1\n")
    other = add(tree, 0, 2.0, code="def solve(x):\n    return [y * 2 for y in range(x)]\n")
    add(tree, original, None, evaluations=7, code="def solve(x):\n    raise RuntimeError()\n")
    copy_id = add(
        tree,
        other,
        1.0,
        op="Crossover",
        reference=original,
        code="# identical normalized AST\ndef solve(x):\n    return x + 1\n",
    )
    assert representatives(tree, cfg) == (original, other)
    assert set(parent_probabilities(tree, topology_config=cfg)) == {original, other}
    assert copy_id not in eligible_reference_pool(tree, other, topology_config=cfg)
    assert tree.nodes[original].direct_evaluations == 7

    plan = plan_simplification(tree, cfg)
    assert copy_id not in plan.kept_ids
    new, _ = reflect_and_apply(tree, plan, model=None)
    assert set(parent_probabilities(new, topology_config=cfg)) == {original, other}
    assert new.nodes[original].direct_evaluations == 7
    assert new.nodes[other].experience


def test_region_size_buffers_inactive_nodes_without_leaking_them_into_sampling():
    cfg = TopologyConfig(max_representatives=8, cover_distance=0.15, region_size=4)
    tree = TreePopulation()
    fork = add(tree, 0, 10.0, code="def solve(x):\n    return x + 10\n")
    m1 = add(tree, fork, 9.0, code="def solve(x):\n    return x + 9\n")
    m2 = add(tree, m1, 8.0, code="def solve(x):\n    return x + 8\n")
    branch_on_m1 = add(tree, m1, None, code="def solve(x):\n    raise ValueError()\n")
    tip = add(tree, m2, 1.0, code="def solve(x):\n    return x + 1\n")  # Covers fork, m1, m2 (d == 0)
    other = add(tree, fork, 2.0, code="def solve(x):\n    return sum(i * 3 for i in range(x))\n")
    worse_clone = add(tree, other, 20.0, code="def solve(x):\n    return sum(i * 5 for i in range(x))\n")

    # Representatives are strictly (tip, other); fork is LCA; m1, m2, branch_on_m1 (3 < 4) and worse_clone (1 < 4) are buffered!
    assert representatives(tree, cfg) == (tip, other)
    assert set(parent_probabilities(tree, topology_config=cfg)) == {tip, other}
    assert eligible_reference_pool(tree, tip, topology_config=cfg) == [other]

    plan = plan_simplification(tree, cfg)
    assert not plan.path_contractions and not plan.subtree_prunings and not plan.triggers
    assert set(plan.kept_ids) == set(tree.nodes)

    # Adding one more failed branch on m2 brings path region to 4 -> triggers PathContraction!
    branch_on_m2 = add(tree, m2, None, code="def solve(x):\n    raise ValueError()\n")
    plan = plan_simplification(tree, cfg)
    assert plan.path_contractions == (PathContraction(fork, tip, (m1, m2), (branch_on_m1, branch_on_m2)),)
    assert not plan.subtree_prunings
    assert worse_clone in plan.kept_ids
    assert plan.triggers == (
        {
            "kind": "reflective_tree_simplification",
            "evaluations": tree.evaluations_used,
            "representative_ids": sorted([tip, other]),
            "fork_ids": [fork],
            "removed_ids": sorted([m1, m2, branch_on_m1, branch_on_m2]),
        },
    )
    tree = apply_simplification(tree, plan)
    edge = tree.edges[tip]
    assert edge.parent_id == fork and edge.contracted and edge.operator is None and edge.reference_id is None

    f1 = add(tree, other, None)
    f2 = add(tree, other, None)
    assert not plan_simplification(tree, cfg).subtree_prunings
    f3 = add(tree, other, None)
    plan = plan_simplification(tree, cfg)
    assert plan.subtree_prunings == (SubtreePruning(other, (worse_clone, f1, f2, f3)),)


def test_experience_transfers_to_node_and_edge_without_inflating_descendant_effort():
    cfg = TopologyConfig(max_representatives=8, cover_distance=0.15, region_size=3)
    tree = TreePopulation()
    a = add(tree, 0, 10.0, code="def solve(x):\n    return x + 10\n")
    b = add(tree, a, 9.0, op="Explore", code="def solve(x):\n    return x + 9\n")
    c = add(tree, b, 8.0, code="def solve(x):\n    return x + 8\n")
    d = add(tree, c, 1.0, code="def solve(x):\n    return x + 1\n")
    branch = add(tree, b, None, evaluations=2)
    other = add(tree, a, 5.0, code="def solve(x):\n    return [k * k for k in range(x)]\n")
    tree.nodes[b].experience = "OLDER_NODE_EVIDENCE"
    tree.edges[c].experience = "OLDER_EDGE_EVIDENCE"
    before = deepcopy(tree)
    plan = plan_simplification(tree, cfg)
    assert plan.kept_ids == (a, d, other)
    (contraction,) = plan.path_contractions
    assert contraction.removed_ids == (b, c) and contraction.branch_ids == (branch,)
    new, _ = reflect_and_apply(tree, plan, model=None)
    assert tree == before
    assert new.nodes[d].direct_evaluations == 0
    assert new.nodes[a].direct_evaluations == 2
    for text in ["OLDER_NODE_EVIDENCE", "OLDER_EDGE_EVIDENCE", "BAD_BRANCH"]:
        assert text in new.edges[d].experience
    assert new.nodes[a].experience == ""
    add(new, d, None, evaluations=2)
    assert new.nodes[d].direct_evaluations == 2


@pytest.mark.parametrize(
    "cfg",
    [
        TopologyConfig(max_representatives=6, cover_distance=0.15, region_size=2),
        TopologyConfig(max_representatives=10, cover_distance=0.20, region_size=4),
    ],
)
@pytest.mark.parametrize("seed", range(8))
def test_budget_and_idempotence_survive_repeated_simplification_against_original_event_history(cfg, seed):
    rng = random.Random(seed)
    tree = TreePopulation()
    events = []
    saw_path = saw_subtree = False
    for step in range(120):
        feasible = [i for i, node in tree.nodes.items() if node.fitness is not None]
        parent = rng.choice(feasible) if feasible else 0
        evaluations = rng.choice([1, 2])
        op = "Refine"
        reference = None
        code = None
        if not parent:
            fit = rng.random()
            code = distinct_code(step)
        else:
            roll = rng.random()
            if roll < 0.15:
                fit = None
                code = "def solve(x):\n    raise RuntimeError()\n"
            elif roll < 0.35:
                # Near/constant variant of parent (same AST shingles -> covered by whichever is better!)
                donor = tree.nodes[parent]
                fit = donor.fitness + rng.uniform(-0.2, 0.2)
                code = f"# tweak {step}\n{donor.code}\n"
            else:
                fit = rng.random()
                code = distinct_code(step)
            other_feasible = [i for i in feasible if i != parent]
            if other_feasible and rng.random() < 0.25:
                op = "Crossover"
                reference = rng.choice(other_feasible)
            else:
                op = rng.choice(["Refine", "Explore"])
        i = add(tree, parent, fit, op=op, reference=reference, evaluations=evaluations, code=code)
        events.append((parent, evaluations))
        reps_before = representatives(tree, cfg)
        probs_before = parent_probabilities(tree, topology_config=cfg)
        plan = plan_simplification(tree, cfg)
        saw_path |= bool(plan.path_contractions)
        saw_subtree |= bool(plan.subtree_prunings)
        for c in plan.path_contractions:
            assert len(c.removed_ids) + len(c.branch_ids) >= cfg.region_size
        for b in plan.subtree_prunings:
            assert len(b.removed_ids) >= cfg.region_size
        removed = [nid for c in plan.path_contractions for nid in (*c.removed_ids, *c.branch_ids)]
        removed += [nid for b in plan.subtree_prunings for nid in b.removed_ids]
        assert len(removed) == len(set(removed))
        assert set(removed) == set(tree.nodes) - set(plan.kept_ids)
        best = champion(tree)
        if best is not None:
            assert best in plan.kept_ids and best == reps_before[0]
        tree = apply_simplification(tree, plan)
        # Simplification NEVER alters representatives or parent probabilities!
        assert representatives(tree, cfg) == reps_before
        assert parent_probabilities(tree, topology_config=cfg) == pytest.approx(probs_before)
        again = plan_simplification(tree, cfg)
        assert not again.path_contractions and not again.subtree_prunings and not again.triggers
        assert again.kept_ids == tuple(sorted(tree.nodes))
        assert tree.evaluations_used == sum(cost for _, cost in events)
        for nid, node in tree.nodes.items():
            assert node.direct_evaluations == sum(e for p, e in events if p == nid)
            assert tree.edges[nid].parent_id == 0 or tree.edges[nid].parent_id in tree.nodes
        assert all(a in tree.nodes and b in tree.nodes for a, b in tree.used_crossover_pairs)
        assert deserialize_tree(json.loads(json.dumps(asdict(tree)))) == tree
    assert saw_path and saw_subtree


def test_reference_quality_uses_equal_ranks_for_ties_and_excludes_parent_first():
    top = TopologyConfig(max_representatives=24, cover_distance=0.0)
    tree = TreePopulation()
    for _ in range(4):
        add(tree, 0, 1)
    assert eligible_reference_pool(tree, 1, topology_config=top) == [2, 3, 4]
    tree = TreePopulation()
    add(tree, 0, 1)
    add(tree, 0, 10)
    assert eligible_reference_pool(tree, 1, topology_config=top) == [2]


@pytest.mark.parametrize("span", [0, -1, float("nan"), float("inf")])
def test_rank_span_is_finite_and_positive(span):
    with pytest.raises(ValueError):
        AllocationConfig(rank_half_life=span)


@pytest.mark.parametrize("field", ["max_representatives", "region_size"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_integer_topology_parameters_are_positive_ints(field, value):
    with pytest.raises(ValueError, match=field):
        TopologyConfig(**{field: value})


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf"), True, "0.15"])
def test_cover_distance_must_be_finite_number_in_unit_interval(value):
    with pytest.raises(ValueError, match="cover_distance"):
        TopologyConfig(cover_distance=value)  # type: ignore[arg-type]


def test_three_round_reflection_reuses_old_notes_and_reaches_all_operators():
    cfg = TopologyConfig(max_representatives=8, cover_distance=0.15, region_size=1)
    t = TreePopulation()
    a = add(t, 0, 10.0, code="def solve(x):\n    return x + 10\n")
    other = add(t, a, 5.0, code="def solve(x):\n    return sum(i * i for i in range(x))\n")
    tip = a
    prompts = []

    class Notes:
        def complete(self, prompt, **kwargs):
            prompts.append(prompt)
            marker = "EDGE" if "[Path Evidence]" in prompt else "NODE"
            return json.dumps({"experience": f"{marker}_NOTE_{round_number}"})

    for round_number in range(3):
        mid = add(t, tip, 9.0 - round_number, code=f"def solve(x):\n    return x + {9 - round_number}\n")
        tip = add(t, mid, 1.0 / (round_number + 1), code=f"def solve(x):\n    return x + {round_number + 1}\n")
        add(t, mid, None)
        add(t, a, None)
        plan = plan_simplification(t, cfg)
        assert plan.path_contractions and plan.subtree_prunings
        prompt_start = len(prompts)
        t, _ = reflect_and_apply(t, plan, model=Notes())
        if round_number:
            used = "\n".join(prompts[prompt_start:])
            assert f"EDGE_NOTE_{round_number - 1}" in used
            assert f"NODE_NOTE_{round_number - 1}" in used
        for op, reference in [("Refine", None), ("Explore", None), ("Crossover", other)]:
            context, _, _, _ = build_evolution_context(t, tip, op, reference)
            assert f"EDGE_NOTE_{round_number}" in context
            assert f"NODE_NOTE_{round_number}" in context
        t = deserialize_tree(json.loads(json.dumps(asdict(t))))
