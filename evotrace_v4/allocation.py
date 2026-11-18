"""全树可行节点上的父代预算分配、交叉参考抽样与探索异质概览选择。"""

from collections.abc import Sequence
from dataclasses import dataclass
import math
import random

from .models import Operator, TreePopulation
from .topology import (
    TopologyConfig,
    code_distance,
    distinct_individuals,
    feasible_individuals,
    identity,
    ranks,
    representatives,
)

OPERATORS: tuple[Operator, ...] = ("Refine", "Crossover", "Explore")
OPERATOR_WEIGHTS = (0.50, 0.30, 0.20)


@dataclass(frozen=True)
class AllocationConfig:
    rank_half_life: float = 6.0
    explore_references: int = 6

    def __post_init__(self) -> None:
        if not math.isfinite(self.rank_half_life) or self.rank_half_life <= 0:
            raise ValueError("rank_half_life must be finite and positive")
        if type(self.explore_references) is not int or self.explore_references < 0:
            raise ValueError("explore_references must be an integer >= 0")


def parent_probabilities(
    tree: TreePopulation,
    config: AllocationConfig = AllocationConfig(),
) -> dict[int, float]:
    """在全树存活可行节点 V_feas 上归一化 2**(-r(p) / h) / (1 + n_p)。"""
    feasible = feasible_individuals(tree)
    if not feasible:
        return {}
    rank = ranks(tree, feasible)
    best = min(rank.values())
    weights = {
        i: 2.0 ** (-(rank[i] - best) / config.rank_half_life) / (1 + tree.nodes[i].direct_evaluations)
        for i in feasible
    }
    total = math.fsum(weights.values())
    return {i: w / total for i, w in weights.items()}


def select_parent(
    tree: TreePopulation,
    rng: random.Random,
    config: AllocationConfig = AllocationConfig(),
) -> int:
    probs = parent_probabilities(tree, config)
    if not probs:
        raise ValueError("TreePopulation has no feasible algorithms to select from")
    return rng.choices(list(probs), weights=list(probs.values()), k=1)[0]


def reference_probabilities(
    tree: TreePopulation,
    parent_id: int,
    config: AllocationConfig = AllocationConfig(),
    topology_config: TopologyConfig = TopologyConfig(),
    *,
    reps: Sequence[int] | None = None,
) -> dict[int, float]:
    """在 S \\ {p}（为空时退至 V_dist \\ {p}）中联合名次衰减与 AST 结构距离抽样异构交叉参考。"""
    parent_code = tree.nodes[parent_id].code
    parent_ident = identity(parent_code)
    core = representatives(tree, topology_config) if reps is None else reps
    pool = [i for i in core if i != parent_id and identity(tree.nodes[i].code) != parent_ident]
    if not pool:
        pool = [
            i for i in sorted(distinct_individuals(tree))
            if i != parent_id and identity(tree.nodes[i].code) != parent_ident
        ]
    if not pool:
        return {}
    rank = ranks(tree, pool)
    best = min(rank.values())
    quality = {i: 2.0 ** (-(rank[i] - best) / config.rank_half_life) for i in sorted(rank)}
    weights = {i: quality[i] * code_distance(parent_code, tree.nodes[i].code) for i in quality}
    total = math.fsum(weights.values())
    if total <= 0.0:
        weights = quality
        total = math.fsum(weights.values())
    return {i: w / total for i, w in weights.items()}


def select_explore_references(
    tree: TreePopulation,
    parent_id: int,
    config: AllocationConfig = AllocationConfig(),
    topology_config: TopologyConfig = TopologyConfig(),
    *,
    reps: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """在 S \\ {p} 非空时先取最优代表，再按相对 p 与已选集合的最大最小 AST 距离选出至多 M 个代表。"""
    if config.explore_references == 0:
        return ()
    parent_ident = identity(tree.nodes[parent_id].code)
    core = representatives(tree, topology_config) if reps is None else reps
    pool = [i for i in core if i != parent_id and identity(tree.nodes[i].code) != parent_ident]
    if not pool:
        return ()
    chosen = [min(pool, key=lambda i: (tree.nodes[i].fitness, i))]  # type: ignore[arg-type]
    while len(chosen) < min(config.explore_references, len(pool)):
        anchors = (parent_id, *chosen)
        chosen.append(max(
            (i for i in pool if i not in chosen),
            key=lambda i: (
                min(code_distance(tree.nodes[j].code, tree.nodes[i].code) for j in anchors),
                -float(tree.nodes[i].fitness),  # type: ignore[arg-type]
                -i,
            ),
        ))
    return tuple(chosen)


def allocate_step(
    tree: TreePopulation,
    rng: random.Random,
    config: AllocationConfig = AllocationConfig(),
    topology_config: TopologyConfig = TopologyConfig(),
) -> tuple[int, Operator, int | None, tuple[int, ...]]:
    """先在全树可行节点上抽样主父代 p，再按先验选择算子并配对交叉参考或探索概览代表。"""
    parent_id = select_parent(tree, rng, config)
    operator = rng.choices(OPERATORS, weights=OPERATOR_WEIGHTS, k=1)[0]
    if operator == "Refine":
        return parent_id, "Refine", None, ()
    reps = representatives(tree, topology_config)
    if operator == "Crossover":
        ref_probs = reference_probabilities(tree, parent_id, config, topology_config, reps=reps)
        if not ref_probs:
            return parent_id, "Refine", None, ()
        ref_id = rng.choices(list(ref_probs), weights=list(ref_probs.values()), k=1)[0]
        return parent_id, "Crossover", ref_id, ()
    return parent_id, "Explore", None, select_explore_references(tree, parent_id, config, topology_config, reps=reps)
