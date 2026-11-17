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
    select_explore_references,
    select_parent,
)
from evotrace_v3.context import _path_from_root, build_evolution_context
from evotrace_v3.models import Edge, Node, TreePopulation
from evotrace_v3.reflection import build_reflection_prompt, reflect_and_apply
from evotrace_v3.topology import (
    PathContraction,
    SubtreePruning,
    TopologyConfig,
    apply_simplification,
    code_distance,
    plan_simplification,
    ranks,
    representatives,
)


class Notes:
    def __init__(self, text="NOTE"):
        self.text = text

    def complete(self, prompt, **kwargs):
        return json.dumps({"experience": self.text})


class Down:
    def complete(self, prompt, **kwargs):
        raise TimeoutError("endpoint down")


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


def add(tree, parent, fit, *, op="Refine", reference=None, evaluations=1, code=None, nid=None):
    i = nid if nid is not None else max(tree.nodes, default=0) + 1
    src = code if code is not None else distinct_code(i - 1)
    tree.add(
        Node(i, src, f"Idea {i}", fit, failure="BAD_BRANCH" if fit is None else None),
        Edge(parent, i, op if parent else None, reference),
        evaluations=evaluations,
    )
    return i


def removed_in(regions):
    return [
        nid for r in regions
        for nid in ((*r.removed_ids, *r.branch_ids) if isinstance(r, PathContraction) else r.removed_ids)
    ]


def test_population_ranks_ignore_units_and_respect_ties():
    tree = TreePopulation()
    for fit in [-20, -20, -10, 0]:
        add(tree, 0, fit)
    assert ranks(tree, tree.nodes) == {1: 0.5, 2: 0.5, 3: 2, 4: 3}
    top = TopologyConfig(max_representatives=24, cover_distance=0.0)
    before = select_parent(tree, random.Random(4), topology_config=top)
    for node in tree.nodes.values():
        node.fitness = 300 * node.fitness + 123
    assert select_parent(tree, random.Random(4), topology_config=top) == before
    for node in tree.nodes.values():
        node.fitness = 10
    assert set(ranks(tree, tree.nodes).values()) == {1.5}
    tree = TreePopulation()
    add(tree, 0, 10)
    assert ranks(tree, tree.nodes) == {1: 0}


def test_quality_pressure_and_direct_effort_have_explicit_odds():
    top = TopologyConfig(max_representatives=32, cover_distance=0.0)
    tree = TreePopulation()
    for fit in range(21):
        add(tree, 0, fit)
    probabilities = parent_probabilities(tree, topology_config=top)
    assert probabilities[1] / probabilities[11] == pytest.approx(2)
    assert probabilities[1] / probabilities[21] == pytest.approx(4)
    add(tree, 1, None)
    probabilities = parent_probabilities(tree, topology_config=top)
    assert probabilities[1] == pytest.approx(probabilities[11])
    improved = add(tree, 1, -1)
    assert tree.nodes[improved].direct_evaluations == 0
    probabilities = parent_probabilities(tree, topology_config=top)
    assert probabilities[improved] > probabilities[1] > 0
    assert probabilities[21] > 0


def test_ties_singleton_and_empty_sampling():
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

    assert select_parent(tree, Capture(), topology_config=top) == first
    parent_id, op, _, _ = allocate_step(tree, random.Random(2), topology_config=top)
    assert parent_id in {first, second} and op in {"Refine", "Explore", "Crossover"}


@pytest.mark.parametrize("span", [1e-300, 1e300])
def test_extreme_rank_span_and_ties_stay_finite(span):
    top = TopologyConfig(max_representatives=24, cover_distance=0.0)
    tree = TreePopulation()
    for fit in [0, 0, 1]:
        add(tree, 0, fit)
    probabilities = parent_probabilities(tree, AllocationConfig(rank_half_life=span), top)
    assert sum(probabilities.values()) == pytest.approx(1)
    assert all(0 <= probability <= 1 for probability in probabilities.values())
    assert probabilities[1] == probabilities[2] > 0
    for node in tree.nodes.values():
        node.fitness = 1
    assert list(parent_probabilities(tree, AllocationConfig(rank_half_life=span), top).values()) == [1 / 3] * 3


def test_representative_cover_collapses_clones_supersedes_constants_and_preserves_lca_forks():
    cfg = TopologyConfig(max_representatives=4, cover_distance=0.15, region_size=2)
    tree = TreePopulation()
    champ = add(tree, 0, 8.30, code="def solve(x):\n    return x + 10\n")
    for k in range(25):
        add(tree, champ, 8.31 + 0.01 * k, code=f"def solve(x):\n    return x + {20 + k}\n")
    rival = add(tree, 0, 8.40, code="def solve(x):\n    return sum(i * i for i in range(x))\n")

    assert representatives(tree, cfg) == (champ, rival)
    probs = parent_probabilities(tree, topology_config=cfg)
    assert set(probs) == {champ, rival} and probs[rival] > probs[champ]

    novel = add(
        tree,
        champ,
        8.35,
        op="Refine",
        code="def solve(x):\n    acc = 1\n    for idx in range(1, x + 1):\n        acc = (acc * 31) ^ idx\n    return acc\n",
    )
    assert code_distance(tree.nodes[champ].code, tree.nodes[novel].code) >= 0.15
    assert representatives(tree, cfg) == (champ, novel, rival)
    assert tree.nodes[novel].direct_evaluations == 0
    probs_after = parent_probabilities(tree, topology_config=cfg)
    assert probs_after[novel] > probs_after[champ]

    better_const = add(tree, champ, 8.25, op="Refine", code="def solve(x):\n    return x + 5\n")
    assert code_distance(tree.nodes[champ].code, tree.nodes[better_const].code) == 0.0
    assert representatives(tree, cfg) == (better_const, novel, rival)
    assert champ not in parent_probabilities(tree, topology_config=cfg)
    # `champ` is retained as the LCA fork connecting `better_const` and `novel`.
    after = apply_simplification(tree, plan_simplification(tree, cfg))
    assert champ in after.nodes and champ not in representatives(after, cfg)


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

    regions = plan_simplification(tree, cfg)
    assert copy_id in removed_in(regions)
    new, _ = reflect_and_apply(tree, regions, model=Notes(), problem="[Problem]")
    assert set(parent_probabilities(new, topology_config=cfg)) == {original, other}
    assert new.nodes[original].direct_evaluations == 7
    assert new.nodes[other].experience == "NOTE"


def test_region_size_buffers_inactive_nodes_without_leaking_them_into_sampling():
    cfg = TopologyConfig(max_representatives=8, cover_distance=0.15, region_size=4)
    tree = TreePopulation()
    fork = add(tree, 0, 10.0, code="def solve(x):\n    return x + 10\n")
    m1 = add(tree, fork, 9.0, code="def solve(x):\n    return x + 9\n")
    m2 = add(tree, m1, 8.0, code="def solve(x):\n    return x + 8\n")
    branch_on_m1 = add(tree, m1, None, code="def solve(x):\n    raise ValueError()\n")
    tip = add(tree, m2, 1.0, code="def solve(x):\n    return x + 1\n")
    other = add(tree, fork, 2.0, code="def solve(x):\n    return sum(i * 3 for i in range(x))\n")
    worse_clone = add(tree, other, 20.0, code="def solve(x):\n    return sum(i * 5 for i in range(x))\n")

    assert representatives(tree, cfg) == (tip, other)
    assert set(parent_probabilities(tree, topology_config=cfg)) == {tip, other}
    assert eligible_reference_pool(tree, tip, topology_config=cfg) == [other]
    assert plan_simplification(tree, cfg) == ()

    branch_on_m2 = add(tree, m2, None, code="def solve(x):\n    raise ValueError()\n")
    regions = plan_simplification(tree, cfg)
    assert regions == (PathContraction(fork, tip, (m1, m2), (branch_on_m1, branch_on_m2)),)
    tree = apply_simplification(tree, regions)
    assert worse_clone in tree.nodes and set(removed_in(regions)).isdisjoint(tree.nodes)
    edge = tree.edges[tip]
    assert edge.parent_id == fork and edge.contracted and edge.operator is None and edge.reference_id is None

    f1 = add(tree, other, None)
    f2 = add(tree, other, None)
    assert plan_simplification(tree, cfg) == ()
    f3 = add(tree, other, None)
    assert plan_simplification(tree, cfg) == (SubtreePruning(other, (worse_clone, f1, f2, f3)),)


def test_experience_transfers_to_node_and_edge_without_inflating_descendant_effort():
    cfg = TopologyConfig(max_representatives=8, cover_distance=0.15, region_size=3)
    tree = TreePopulation()
    a = add(tree, 0, 10.0, code="def solve(x):\n    return x + 10\n")
    b = add(tree, a, 9.0, code="def solve(x):\n    return x + 9\n")
    c = add(tree, b, 8.0, code="def solve(x):\n    return x + 8\n")
    d = add(tree, c, 1.0, code="def solve(x):\n    return x + 1\n")
    branch = add(tree, b, None, evaluations=2)
    other = add(tree, a, 5.0, code="def solve(x):\n    return [k * k for k in range(x)]\n")
    tree.nodes[b].experience = "OLDER_NODE_EVIDENCE"
    tree.edges[c].experience = "OLDER_EDGE_EVIDENCE"
    before = deepcopy(tree)
    regions = plan_simplification(tree, cfg)
    (contraction,) = regions
    assert contraction.removed_ids == (b, c) and contraction.branch_ids == (branch,)
    prompt, _ = build_reflection_prompt(tree, contraction, "[Problem]")
    assert "BAD_BRANCH" in prompt
    new, _ = reflect_and_apply(tree, regions, model=Down(), problem="[Problem]")
    assert tree == before
    assert set(new.nodes) == {a, d, other}
    assert new.nodes[d].direct_evaluations == 0
    assert new.nodes[a].direct_evaluations == 2
    assert new.edges[d].experience == "OLDER_NODE_EVIDENCE OLDER_EDGE_EVIDENCE"
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
        if not parent:
            fit = rng.random()
            code = distinct_code(step)
        else:
            roll = rng.random()
            if roll < 0.15:
                fit = None
                code = "def solve(x):\n    raise RuntimeError()\n"
            elif roll < 0.35:
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
            elif rng.random() < 0.5:
                parent, op = 0, "Explore"
        add(tree, parent, fit, op=op, reference=reference, evaluations=evaluations, code=code, nid=step + 1)
        events.append((parent, evaluations))
        reps_before = representatives(tree, cfg)
        probs_before = parent_probabilities(tree, topology_config=cfg)
        regions = plan_simplification(tree, cfg)
        saw_path |= any(isinstance(r, PathContraction) for r in regions)
        saw_subtree |= any(isinstance(r, SubtreePruning) for r in regions)
        for r in regions:
            size = len(r.removed_ids) + (len(r.branch_ids) if isinstance(r, PathContraction) else 0)
            assert size >= cfg.region_size
        removed = removed_in(regions)
        assert len(removed) == len(set(removed))
        best = min((i for i in tree.nodes if tree.nodes[i].fitness is not None),
                   key=lambda i: (tree.nodes[i].fitness, i), default=None)
        tree = apply_simplification(tree, regions)
        if best is not None:
            assert best in tree.nodes and best == reps_before[0]
        assert representatives(tree, cfg) == reps_before
        assert parent_probabilities(tree, topology_config=cfg) == pytest.approx(probs_before)
        assert plan_simplification(tree, cfg) == ()
        for nid, node in tree.nodes.items():
            assert node.direct_evaluations == sum(e for p, e in events if p == nid)
            assert tree.edges[nid].parent_id == 0 or tree.edges[nid].parent_id in tree.nodes
        assert all(a in tree.nodes and b in tree.nodes for a, b in tree.used_crossover_pairs)
        assert TreePopulation.from_dict(json.loads(json.dumps(asdict(tree)))) == tree
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


def test_three_round_reflection_reuses_old_notes_and_reaches_operators_that_show_them():
    cfg = TopologyConfig(max_representatives=8, cover_distance=0.15, region_size=1)
    t = TreePopulation()
    a = add(t, 0, 10.0, code="def solve(x):\n    return x + 10\n")
    other = add(t, a, 5.0, code="def solve(x):\n    return sum(i * i for i in range(x))\n")
    tip = a
    prompts = []

    class RoundNotes:
        def complete(self, prompt, **kwargs):
            prompts.append(prompt)
            marker = "EDGE" if "developed from" in prompt else "NODE"
            return json.dumps({"experience": f"{marker}_NOTE_{round_number}"})

    for round_number in range(3):
        mid = add(t, tip, 9.0 - round_number, code=f"def solve(x):\n    return x + {9 - round_number}\n")
        tip = add(t, mid, 1.0 / (round_number + 1), code=f"def solve(x):\n    return x + {round_number + 1}\n")
        add(t, mid, None)
        add(t, a, None)
        regions = plan_simplification(t, cfg)
        assert any(isinstance(r, PathContraction) for r in regions)
        assert any(isinstance(r, SubtreePruning) for r in regions)
        prompt_start = len(prompts)
        t, _ = reflect_and_apply(t, regions, model=RoundNotes(), problem="[Problem]")
        if round_number:
            used = "\n".join(prompts[prompt_start:])
            assert f"EDGE_NOTE_{round_number - 1}" in used
            assert f"NODE_NOTE_{round_number - 1}" in used
        holders = {i for i, n in t.nodes.items() if f"NODE_NOTE_{round_number}" in n.experience}
        for op, reference in [("Refine", None), ("Explore", None), ("Crossover", other)]:
            ids = select_explore_references(t, tip, topology_config=cfg) if op == "Explore" else ()
            context, _ = build_evolution_context(t, tip, op, reference, explore_ids=ids)
            assert f"EDGE_NOTE_{round_number}" in context
            on_paths = set().union(*(_path_from_root(t, i) for i in (tip, *ids)))
            shown = op != "Explore" or bool(holders & on_paths)
            assert (f"NODE_NOTE_{round_number}" in context) == shown
        t = TreePopulation.from_dict(json.loads(json.dumps(asdict(t))))


def test_explore_references_take_the_best_then_spread_away_from_parent_and_each_other():
    tree = TreePopulation()
    parent = add(tree, 0, 5.0, code="def solve(x):\n    return x + 1\n")
    best = add(tree, 0, 1.0, code="def solve(x):\n    return x + 2\n")
    near = add(tree, 0, 2.0, code="def solve(x):\n    return x * 2 + 1\n")
    far = add(tree, 0, 3.0, code="def solve(x):\n    acc = 0\n    for k in range(x):\n        acc ^= k << 1\n    return acc\n")
    weak = add(tree, 0, 4.0, code="def solve(x):\n    return sorted([x, x - 1])[0]\n")
    top = TopologyConfig(cover_distance=0.0)
    assert select_explore_references(tree, parent, AllocationConfig(explore_references=2), top) == (best, far)
    assert select_explore_references(tree, parent, AllocationConfig(), top) == (best, far, weak, near)
    alone = TreePopulation()
    only = add(alone, 0, 1.0)
    assert select_explore_references(alone, only, AllocationConfig(), top) == ()
