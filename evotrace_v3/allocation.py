"""代表集上的父代抽样、交叉参考抽样与 Explore 已有做法选择。"""

from collections.abc import Sequence
from dataclasses import dataclass
import math
import random

from .models import Operator, ROOT_ID, TreePopulation
from .topology import TopologyConfig, code_distance, ranks, representatives

OPERATORS: tuple[Operator, ...] = ("Refine", "Explore", "Crossover")
OPERATOR_WEIGHTS = (0.45, 0.30, 0.25)


@dataclass(frozen=True)
class AllocationConfig:
    rank_half_life: float = 10.0
    explore_references: int = 6

    def __post_init__(self) -> None:
        if not math.isfinite(self.rank_half_life) or self.rank_half_life <= 0:
            raise ValueError("rank_half_life must be finite and positive")
        if type(self.explore_references) is not int or self.explore_references < 0:
            raise ValueError("explore_references must be an integer >= 0")


def parent_probabilities(
    tree: TreePopulation, config: AllocationConfig = AllocationConfig(),
    topology_config: TopologyConfig = TopologyConfig(), *, reps: Sequence[int] | None = None,
) -> dict[int, float]:
    """在代表集上归一化 2**(-rank / rank_half_life) / (1 + direct_evaluations)。"""
    rank = ranks(tree, representatives(tree, topology_config) if reps is None else reps)
    best = min(rank.values(), default=0.0)
    weights = {
        i: 2 ** (-(rank[i] - best) / config.rank_half_life) / (1 + tree.nodes[i].direct_evaluations)
        for i in sorted(rank)
    }
    total = math.fsum(weights.values())
    return {i: w / total for i, w in weights.items()}


def select_parent(
    tree: TreePopulation, rng: random.Random, config: AllocationConfig = AllocationConfig(),
    topology_config: TopologyConfig = TopologyConfig(), *, reps: Sequence[int] | None = None,
) -> int:
    probs = parent_probabilities(tree, config, topology_config, reps=reps)
    if not probs:
        raise ValueError("TreePopulation has no feasible algorithms to select from")
    return rng.choices(list(probs), weights=list(probs.values()), k=1)[0]


def reference_probabilities(
    tree: TreePopulation, parent_id: int, config: AllocationConfig = AllocationConfig(),
    topology_config: TopologyConfig = TopologyConfig(), *, reps: Sequence[int] | None = None,
) -> dict[int, float]:
    """在除主父代外的代表集上归一化 2**(-rank / rank_half_life) * code_distance(parent, r)。"""
    pool = [i for i in (representatives(tree, topology_config) if reps is None else reps) if i != parent_id]
    if not pool:
        return {}
    rank = ranks(tree, pool)
    best = min(rank.values())
    code = tree.nodes[parent_id].code
    quality = {i: 2 ** (-(rank[i] - best) / config.rank_half_life) for i in sorted(rank)}
    weights = {i: quality[i] * code_distance(code, tree.nodes[i].code) for i in quality}
    total = math.fsum(weights.values())
    if total <= 0.0:
        weights = quality
        total = math.fsum(weights.values())
    return {i: w / total for i, w in weights.items()}


def select_explore_references(
    tree: TreePopulation, config: AllocationConfig = AllocationConfig(),
    topology_config: TopologyConfig = TopologyConfig(), *, reps: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """在代表集上先取适应度最优者，再逐个取与已选个体最小 AST 距离最大者。"""
    pool = list(representatives(tree, topology_config) if reps is None else reps)
    chosen = sorted(pool, key=lambda i: (tree.nodes[i].fitness, i))[:min(1, config.explore_references)]  # type: ignore[arg-type]
    while len(chosen) < min(config.explore_references, len(pool)):
        chosen.append(max(
            (i for i in pool if i not in chosen),
            key=lambda i: (
                min(code_distance(tree.nodes[j].code, tree.nodes[i].code) for j in chosen),
                -float(tree.nodes[i].fitness),  # type: ignore[arg-type]
                -i,
            ),
        ))
    return tuple(chosen)


def allocate_step(
    tree: TreePopulation, rng: random.Random, config: AllocationConfig = AllocationConfig(),
    topology_config: TopologyConfig = TopologyConfig(),
) -> tuple[int, Operator, int | None, tuple[int, ...]]:
    """选择演化算子，并在代表集上分配主父代、交叉参考或 Explore 已有做法集合。"""
    reps = representatives(tree, topology_config)
    if not reps:
        raise ValueError("TreePopulation has no feasible algorithms to select from")
    operator = rng.choices(OPERATORS, weights=OPERATOR_WEIGHTS, k=1)[0]
    if operator == "Explore":
        return ROOT_ID, "Explore", None, select_explore_references(tree, config, topology_config, reps=reps)
    parent_id = select_parent(tree, rng, config, topology_config, reps=reps)
    if operator == "Crossover":
        ref_probs = reference_probabilities(tree, parent_id, config, topology_config, reps=reps)
        if not ref_probs:
            return parent_id, "Refine", None, ()
        ref_id = rng.choices(list(ref_probs), weights=list(ref_probs.values()), k=1)[0]
        return parent_id, "Crossover", ref_id, ()
    return parent_id, "Refine", None, ()
