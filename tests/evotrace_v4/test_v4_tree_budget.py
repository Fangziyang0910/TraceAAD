from copy import deepcopy
from dataclasses import asdict
import json
import random

import pytest

from evotrace_v4.allocation import (
    AllocationConfig,
    allocate_step,
    parent_probabilities,
    reference_probabilities,
    select_explore_references,
    select_parent,
)
from evotrace_v4.models import Edge, Node, ROOT_ID, TreePopulation
from evotrace_v4.reflection import reflect_and_apply
from evotrace_v4.topology import (
    PathContraction,
    SubtreePruning,
    TopologyConfig,
    apply_simplification,
    code_distance,
    feasible_individuals,
    plan_simplification,
    ranks,
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


def add(tree, parent, fit, *, op="Refine", reference=None, evaluations=1, code=None, nid=None):
    i = nid if nid is not None else max(tree.nodes, default=0) + 1
    src = code if code is not None else distinct_code(i - 1)
    tree.add(
        Node(i, src, f"Idea {i}", fit, failure="BAD_BRANCH" if fit is None else None),
        Edge(parent, i, None if parent == ROOT_ID else op, reference),
        evaluations=evaluations,
    )
    return i


def removed_in(regions):
    return [
        nid for r in regions
        for nid in ((*r.removed_ids, *r.branch_ids) if isinstance(r, PathContraction) else r.removed_ids)
    ]


class Notes:
    def __init__(self, text="NOTE"):
        self.text = text

    def complete(self, prompt, **kwargs):
        return json.dumps({"experience": self.text})


def test_tree_node_sampling_keeps_exact_copies_until_pruning_with_half_integer_rank_and_13_29_share():
    """复核设计定稿中的小树算例：5 个同分异构初始算法 + A 的 3 个同码同分子代 -> 平均名次 3.5，A 代码总份额 13/29。"""
    cfg = TopologyConfig(max_representatives=24, cover_distance=0.15, region_size=4)
    tree = TreePopulation()
    init_ids = [add(tree, 0, 10.0, code=distinct_code(k)) for k in range(5)]
    a_id = init_ids[0]
    a_code = tree.nodes[a_id].code
    copies = [add(tree, a_id, 10.0, op="Refine", code=a_code) for _ in range(3)]

    assert representatives(tree, cfg) == tuple(init_ids)
    assert plan_simplification(tree, cfg) == ()  # 3 < m=4, buffered on the tree
    feasible = feasible_individuals(tree)
    assert len(feasible) == 8
    r = ranks(tree, feasible)
    assert set(r.values()) == {3.5}
    assert tree.nodes[a_id].direct_evaluations == 3
    assert all(tree.nodes[c].direct_evaluations == 0 for c in copies)

    probs = parent_probabilities(tree, AllocationConfig(rank_half_life=6.0))
    share_a = probs[a_id] + sum(probs[c] for c in copies)
    assert share_a == pytest.approx(13 / 29)

    # Fourth unretained child under A merges with the three copies into one SubtreePruning(a_id, ...) of size 4.
    fourth = add(tree, a_id, 10.0, op="Explore", code=a_code)
    regions = plan_simplification(tree, cfg)
    assert regions == (SubtreePruning(a_id, (*copies, fourth)),)
    after, _ = reflect_and_apply(tree, regions, model=Notes(), problem="[Problem]")
    assert after.nodes[a_id].direct_evaluations == 4  # Preserved without reset or grandchild inflation.
    assert set(feasible_individuals(after)) == set(init_ids)


def test_stepping_stones_and_explore_children_compete_for_evaluations_before_regional_pruning():
    cfg = TopologyConfig(max_representatives=2, cover_distance=0.15, region_size=4)
    tree = TreePopulation()
    champ = add(tree, 0, 8.01, code="def solve(x):\n    return x + 1\n")
    rival = add(tree, 0, 8.10, code="def solve(x):\n    return sum(i * i for i in range(x))\n")
    # Stepping stone slightly worse than champ with AST distance 0 (< 0.15) is excluded from S,
    # YET remains in V_feas and competes for parent evaluations!
    stone = add(tree, champ, 8.02, op="Refine", code="def solve(x):\n    return x + 2\n")
    explore_child = add(
        tree,
        champ,
        8.30,
        op="Explore",
        code="def solve(x):\n    acc = 0\n    for k in range(x):\n        acc ^= (k << 1)\n    return acc\n",
    )
    assert representatives(tree, cfg) == (champ, rival)
    probs = parent_probabilities(tree, AllocationConfig(rank_half_life=6.0))
    assert set(probs) == {champ, rival, stone, explore_child}
    assert probs[stone] > probs[champ]  # Rank 1 with n=0 beats Rank 0 with n=2!
    assert probs[explore_child] > 0.0
    assert plan_simplification(tree, cfg) == ()


def test_merged_single_leaf_siblings_and_contracted_regressing_refine_to_root():
    """复核两条关键拓扑规则：
    1) 同一骨架节点下的 4 个单叶落选旁支合并为一个子树区一次剪除；
    2) 目标值为 2 的祖先因结构覆盖出局，其目标值为 3 的 Refine 后代因结构不同入选 S 并收缩重连至虚拟根 0。
    """
    cfg = TopologyConfig(max_representatives=4, cover_distance=0.15, region_size=4)
    tree = TreePopulation()
    dominant = add(tree, 0, 1.0, code="def solve(x):\n    return x + 1\n")
    # Ancestor u (fitness=2) has identical AST structure to dominant (distance 0 < 0.15), so u is covered!
    anc_u = add(tree, 0, 2.0, code="def solve(x):\n    return x + 2\n")
    # Descendant w (fitness=3 > 2, worse than anc_u!) comes from Refine on anc_u, with distinct AST >= 0.15.
    desc_w = add(tree, anc_u, 3.0, op="Refine", code="def solve(x):\n    return sum(i * i for i in range(x))\n")
    assert code_distance(tree.nodes[dominant].code, tree.nodes[anc_u].code) < 0.15
    assert code_distance(tree.nodes[dominant].code, tree.nodes[desc_w].code) >= 0.15
    assert representatives(tree, cfg) == (dominant, desc_w)

    # Add 3 side leaves on anc_u so PathContraction(0, desc_w) reaches size 1 + 3 = 4,
    # AND 4 single-leaf rejected children directly under dominant so SubtreePruning(dominant) reaches size 4!
    u_sides = tuple(add(tree, anc_u, None) for _ in range(3))
    dom_leaves = tuple(add(tree, dominant, 9.0 + k, code=f"def solve(x):\n    return x + {100 + k}\n") for k in range(4))

    regions = plan_simplification(tree, cfg)
    assert PathContraction(ROOT_ID, desc_w, (anc_u,), u_sides) in regions
    assert SubtreePruning(dominant, dom_leaves) in regions

    after = apply_simplification(tree, regions)
    assert set(after.nodes) == {dominant, desc_w}
    assert after.edges[desc_w].parent_id == ROOT_ID
    assert after.edges[desc_w].contracted is True and after.edges[desc_w].operator is None
    assert plan_simplification(after, cfg) == ()


def test_five_close_initial_algorithms_handle_empty_core_complement_in_explore_and_crossover():
    """结构距离均为零时探索没有参考，交叉仍可使用源码不同的候选。"""
    cfg = TopologyConfig(max_representatives=24, cover_distance=0.15, region_size=4)
    alloc = AllocationConfig(rank_half_life=6.0, explore_references=6)
    tree = TreePopulation()
    inits = [add(tree, 0, float(k + 1), code=f"def solve(x):\n    return x + {k + 1}\n") for k in range(5)]
    p = inits[0]
    assert representatives(tree, cfg) == (p,)
    assert select_explore_references(tree, p, random.Random(0), alloc) == ()
    # Changing the parent leaves all candidate distances at zero.
    assert select_explore_references(tree, inits[1], random.Random(0), alloc) == ()
    ref_probs = reference_probabilities(tree, p, alloc, cfg)
    assert set(ref_probs) == set(inits[1:])
    assert sum(ref_probs.values()) == pytest.approx(1.0)

    # When all feasible nodes share identical standardized source, Crossover falls back to Refine.
    clone_tree = TreePopulation()
    only = add(clone_tree, 0, 1.0, code="def solve(x):\n    return x + 1\n")
    twin = add(clone_tree, only, 1.0, code="# comment\ndef solve(x):\n    return x + 1\n")
    assert reference_probabilities(clone_tree, only, alloc, cfg) == {}
    assert reference_probabilities(clone_tree, twin, alloc, cfg) == {}
    for seed in range(20):
        parent_id, op, ref_id, _ = allocate_step(clone_tree, random.Random(seed), alloc, cfg)
        assert parent_id in {only, twin}
        assert op in {"Refine", "Explore"} and ref_id is None


def test_explore_samples_all_living_sources_without_replacement():
    tree = TreePopulation()
    ids = [add(tree, 0, float(k + 1), code=distinct_code(k)) for k in range(9)]
    p = ids[0]
    add(tree, p, 0.5, code="# parent copy\n" + tree.nodes[p].code)
    add(tree, p, 0.8, code="# reference copy\n" + tree.nodes[ids[1]].code)
    add(tree, p, None, code=distinct_code(12))
    eligible = set(ids[1:])
    cfg = TopologyConfig(max_representatives=2)
    core = set(representatives(tree, cfg))
    assert eligible - core
    alloc = AllocationConfig(explore_references=5)
    seen = set()
    for seed in range(30):
        chosen = select_explore_references(tree, p, random.Random(seed), alloc)
        assert len(chosen) == len(set(chosen)) == 5
        assert set(chosen) <= eligible
        assert chosen == select_explore_references(tree, p, random.Random(seed), alloc)
        seen.update(chosen)
    assert seen == eligible
    wide = AllocationConfig(explore_references=24)
    assert set(select_explore_references(tree, p, random.Random(0), wide)) == eligible

    operators = set()
    for seed in range(50):
        _, operator, reference, cards = allocate_step(tree, random.Random(seed), alloc, cfg)
        operators.add(operator)
        if operator == "Explore":
            assert set(cards) - core
        if operator == "Crossover":
            assert reference in core
    assert {"Explore", "Crossover"} <= operators


def test_explore_reference_weights_use_pool_ranks_and_nearest_distance():
    """复核每轮权重：池内名次衰减乘以到 {p} 与已选代表的最小 AST 距离。"""
    alloc = AllocationConfig(rank_half_life=6.0, explore_references=2)
    tree = TreePopulation()
    ids = [add(tree, 0, float(k + 1), code=distinct_code(k)) for k in range(6)]
    p = ids[2]  # A parent in the middle of the ranking: pool ranks must exclude it.

    class Recorder:
        def __init__(self):
            self.rounds = []

        def choices(self, population, weights, k=1):
            self.rounds.append(dict(zip(population, weights)))
            return [population[-1]]

    rng = Recorder()
    chosen = select_explore_references(tree, p, rng, alloc)
    pool = [i for i in ids if i != p]
    rank = ranks(tree, pool)
    code = {i: tree.nodes[i].code for i in ids}
    first, second = rng.rounds
    best = min(rank.values())
    assert first == pytest.approx({i: 2 ** (-(rank[i] - best) / 6.0) * code_distance(code[p], code[i]) for i in pool})
    pick = chosen[0]
    rest = [i for i in pool if i != pick]
    best = min(rank[i] for i in rest)
    assert second == pytest.approx({
        i: 2 ** (-(rank[i] - best) / 6.0) * min(code_distance(code[p], code[i]), code_distance(code[pick], code[i]))
        for i in rest
    })


@pytest.mark.parametrize(
    "cfg",
    [
        TopologyConfig(max_representatives=6, cover_distance=0.15, region_size=2),
        TopologyConfig(max_representatives=24, cover_distance=0.15, region_size=4),
    ],
)
@pytest.mark.parametrize("seed", range(6))
def test_node_upper_bound_idempotence_and_effort_accounting_across_random_evolutions(cfg, seed):
    rng = random.Random(seed)
    alloc = AllocationConfig(rank_half_life=6.0, explore_references=6)
    tree = TreePopulation()
    events: list[tuple[int, int]] = []
    upper_bound = 4 * cfg.max_representatives * cfg.region_size - 2 * cfg.max_representatives - cfg.region_size
    if cfg.max_representatives == 24 and cfg.region_size == 4:
        assert upper_bound == 332

    for step in range(5):
        add(tree, 0, rng.random(), code=distinct_code(step), nid=step + 1)
        events.append((0, 1))

    saw_path = saw_subtree = False
    for step in range(5, 140):
        parent, op, ref, _ = allocate_step(tree, rng, alloc, cfg)
        assert parent > 0  # No evolution operator ever attaches to ROOT_ID!
        evaluations = rng.choice([1, 2])
        roll = rng.random()
        if roll < 0.15:
            fit, code = None, "def solve(x):\n    raise RuntimeError()\n"
        elif roll < 0.40:
            donor = tree.nodes[parent]
            fit = donor.fitness + rng.uniform(-0.2, 0.2)
            code = f"# tweak {step}\n{donor.code}\n"
        else:
            fit, code = rng.random(), distinct_code(step)
        add(tree, parent, fit, op=op, reference=ref, evaluations=evaluations, code=code, nid=step + 1)
        events.append((parent, evaluations))

        reps_before = representatives(tree, cfg)
        regions = plan_simplification(tree, cfg)
        saw_path |= any(isinstance(r, PathContraction) for r in regions)
        saw_subtree |= any(isinstance(r, SubtreePruning) for r in regions)
        removed = removed_in(regions)
        assert len(removed) == len(set(removed))
        tree = apply_simplification(tree, regions)

        assert len(tree.nodes) <= upper_bound
        assert representatives(tree, cfg) == reps_before
        assert plan_simplification(tree, cfg) == ()
        for nid, node in tree.nodes.items():
            assert node.direct_evaluations == sum(e for p, e in events if p == nid)
            assert tree.edges[nid].parent_id == ROOT_ID or tree.edges[nid].parent_id in tree.nodes
        assert TreePopulation.from_dict(json.loads(json.dumps(asdict(tree)))) == tree
    assert saw_path and saw_subtree
