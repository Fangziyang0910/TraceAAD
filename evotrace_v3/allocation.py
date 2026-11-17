"""代表集上的父代抽样、交叉参考轮换与 Explore 参考选择。"""

from collections.abc import Sequence
from dataclasses import dataclass
import math
import random

from .models import Operator, TreePopulation
from .topology import TopologyConfig, code_distance, ranks, representatives

OPERATORS: tuple[Operator, ...] = ("Refine", "Explore", "Crossover")
OPERATOR_WEIGHTS = (0.45, 0.30, 0.25)
REFERENCE_QUALITY_FLOOR = 0.30


@dataclass(frozen=True)
class AllocationConfig:
    rank_half_life: float = 10.0
    explore_references: int = 6

    def __post_init__(self) -> None:
        if not math.isfinite(self.rank_half_life) or self.rank_half_life <= 0:
            raise ValueError("rank_half_life must be finite and positive")


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


def eligible_reference_pool(
    tree: TreePopulation, parent_id: int, topology_config: TopologyConfig = TopologyConfig(),
    *, reps: Sequence[int] | None = None,
) -> list[int]:
    """除主父代外，归一化名次质量 1 - rank / (n - 1)（单候选取 0.5）不低于下限的代表个体。"""
    pool = [i for i in (representatives(tree, topology_config) if reps is None else reps) if i != parent_id]
    rank = ranks(tree, pool)
    return [i for i in pool if (1 - rank[i] / (len(pool) - 1) if len(pool) > 1 else 0.5) >= REFERENCE_QUALITY_FLOOR]


def select_explore_references(
    tree: TreePopulation, parent_id: int, config: AllocationConfig = AllocationConfig(),
    topology_config: TopologyConfig = TopologyConfig(), *, reps: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """除主父代外，先取适应度最优者，再逐个取与主父代及已选个体最小 AST 距离最大者。"""
    pool = [i for i in (representatives(tree, topology_config) if reps is None else reps) if i != parent_id]
    chosen = sorted(pool, key=lambda i: (tree.nodes[i].fitness, i))[:min(1, config.explore_references)]  # type: ignore[arg-type]
    while len(chosen) < min(config.explore_references, len(pool)):
        chosen.append(max(
            (i for i in pool if i not in chosen),
            key=lambda i: (
                min(code_distance(tree.nodes[j].code, tree.nodes[i].code) for j in (parent_id, *chosen)),
                -float(tree.nodes[i].fitness),  # type: ignore[arg-type]
                -i,
            ),
        ))
    return tuple(chosen)


def allocate_step(
    tree: TreePopulation, rng: random.Random, config: AllocationConfig = AllocationConfig(),
    topology_config: TopologyConfig = TopologyConfig(),
) -> tuple[int, Operator, int | None, tuple[int, ...]]:
    """选择主父代、演化算子及 Crossover 参考或 Explore 代表集合。"""
    reps = representatives(tree, topology_config)
    parent_id = select_parent(tree, rng, config, topology_config, reps=reps)
    operator = rng.choices(OPERATORS, weights=OPERATOR_WEIGHTS, k=1)[0]
    if operator == "Crossover":
        pool = eligible_reference_pool(tree, parent_id, topology_config, reps=reps)
        if not pool:
            return parent_id, "Refine", None, ()
        used = set(tree.used_crossover_pairs)
        candidates = [i for i in pool if (parent_id, i) not in used] or pool
        code = tree.nodes[parent_id].code
        ref_id = max(candidates, key=lambda i: (code_distance(code, tree.nodes[i].code), -float(tree.nodes[i].fitness), -i))  # type: ignore[arg-type]
        return parent_id, "Crossover", ref_id, ()
    if operator == "Explore":
        return parent_id, "Explore", None, select_explore_references(tree, parent_id, config, topology_config, reps=reps)
    return parent_id, "Refine", None, ()
