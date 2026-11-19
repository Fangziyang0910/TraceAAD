"""全树可行节点上的父代预算分配、交叉参考抽样与探索代表抽样。"""

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
    explore_references: int = 5

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
    rng: random.Random,
    config: AllocationConfig = AllocationConfig(),
) -> tuple[int, ...]:
    """从全树存活的可行个体中按标准化源码去重，排除主父代源码，抽取至多 M 个代表。

    候选 q 的权重为池内名次衰减 2**(-r(q) / h) 乘以 q 到 {p} 与已选代表的最小 AST 距离；
    名次衰减每轮以剩余候选中的最优名次为基准（不改变抽样分布，只避免极小 h 下的浮点下溢）。
    距离为零的候选不参与抽样，没有正距离候选时提前停止。
    """
    if config.explore_references == 0:
        return ()
    seen = {identity(tree.nodes[parent_id].code)}
    pool = []
    for i in feasible_individuals(tree):
        source = identity(tree.nodes[i].code)
        if source not in seen:
            seen.add(source)
            pool.append(i)
    if not pool:
        return ()
    rank = ranks(tree, pool)
    nearest = {i: code_distance(tree.nodes[parent_id].code, tree.nodes[i].code) for i in pool}
    chosen: list[int] = []
    while len(chosen) < config.explore_references:
        remaining = [i for i in pool if i not in chosen and nearest[i] > 0.0]
        if not remaining:
            break
        best = min(rank[i] for i in remaining)
        weights = [2.0 ** (-(rank[i] - best) / config.rank_half_life) * nearest[i] for i in remaining]
        pick = rng.choices(remaining, weights=weights, k=1)[0]
        chosen.append(pick)
        for i in pool:
            nearest[i] = min(nearest[i], code_distance(tree.nodes[pick].code, tree.nodes[i].code))
    return tuple(chosen)


def allocate_step(
    tree: TreePopulation,
    rng: random.Random,
    config: AllocationConfig = AllocationConfig(),
    topology_config: TopologyConfig = TopologyConfig(),
) -> tuple[int, Operator, int | None, tuple[int, ...]]:
    """先在全树可行节点上抽样主父代 p，再按先验选择算子并配对交叉参考或探索代表。"""
    parent_id = select_parent(tree, rng, config)
    operator = rng.choices(OPERATORS, weights=OPERATOR_WEIGHTS, k=1)[0]
    if operator == "Refine":
        return parent_id, "Refine", None, ()
    if operator == "Crossover":
        ref_probs = reference_probabilities(tree, parent_id, config, topology_config)
        if not ref_probs:
            return parent_id, "Refine", None, ()
        ref_id = rng.choices(list(ref_probs), weights=list(ref_probs.values()), k=1)[0]
        return parent_id, "Crossover", ref_id, ()
    return parent_id, "Explore", None, select_explore_references(tree, parent_id, rng, config)
