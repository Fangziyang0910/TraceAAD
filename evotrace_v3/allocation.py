"""Parent selection, crossover reference and Explore references over the representative set."""

from dataclasses import dataclass
import math
import random

from .models import Operator, TreePopulation
from .topology import TopologyConfig, code_distance, ranks, representatives


@dataclass(frozen=True)
class AllocationConfig:
    rank_half_life: float = 10.0  # Representative rank positions over which quality weight halves.
    refine_weight: float = 0.45
    explore_weight: float = 0.30
    crossover_weight: float = 0.25
    reference_quality_floor: float = 0.30
    explore_references: int = 6  # Explore 展示的代表个体数

    def __post_init__(self) -> None:
        if not math.isfinite(self.rank_half_life) or self.rank_half_life <= 0:
            raise ValueError("rank_half_life must be finite and positive")
        weights = (self.refine_weight, self.explore_weight, self.crossover_weight)
        if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError("Operator weights must be nonnegative and sum to a positive value")
        if not 0 <= self.reference_quality_floor <= 1:
            raise ValueError("reference_quality_floor must be in [0, 1]")


@dataclass(frozen=True)
class AllocationDecision:
    parent_id: int
    operator: Operator
    reference_id: int | None
    parent_probability: float
    reference_distance: float | None = None
    explore_ids: tuple[int, ...] = ()


def parent_probabilities(
    tree: TreePopulation,
    config: AllocationConfig = AllocationConfig(),
    topology_config: TopologyConfig = TopologyConfig(),
) -> dict[int, float]:
    """Normalize 2**(-rank / rank_half_life) / (1 + direct_evaluations) over the representatives."""
    rank = ranks(tree, representatives(tree, topology_config))
    best = min(rank.values(), default=0.0)  # 只做数值平移，归一化后不变
    weights = {i: 2 ** (-(rank[i] - best) / config.rank_half_life) / (1 + tree.nodes[i].direct_evaluations)
               for i in sorted(rank)}
    total = math.fsum(weights.values())
    return {i: weight / total for i, weight in weights.items()}


def select_parent(
    tree: TreePopulation,
    rng: random.Random,
    config: AllocationConfig = AllocationConfig(),
    topology_config: TopologyConfig = TopologyConfig(),
) -> tuple[int, float]:
    probabilities = parent_probabilities(tree, config, topology_config)
    if not probabilities:
        raise ValueError("TreePopulation has no feasible algorithms to select from")
    parent = rng.choices(list(probabilities), weights=list(probabilities.values()), k=1)[0]
    return parent, probabilities[parent]


def eligible_reference_pool(
    tree: TreePopulation,
    parent_id: int,
    config: AllocationConfig = AllocationConfig(),
    topology_config: TopologyConfig = TopologyConfig(),
) -> list[int]:
    """Representatives other than the parent whose rank quality among them is at least ``reference_quality_floor``.

    Rank quality is 1 - rank / (n - 1), and 0.5 for a single candidate.
    """
    pool = [i for i in representatives(tree, topology_config) if i != parent_id]
    rank = ranks(tree, pool)
    quality = {i: 1 - rank[i] / (len(pool) - 1) if len(pool) > 1 else 0.5 for i in pool}
    return [i for i in pool if quality[i] >= config.reference_quality_floor]


def select_crossover_reference(
    tree: TreePopulation,
    parent_id: int,
    config: AllocationConfig = AllocationConfig(),
    topology_config: TopologyConfig = TopologyConfig(),
) -> tuple[int | None, float | None]:
    """Among eligible references, prefer pairs with the parent not yet used, then the largest AST distance."""
    pool = eligible_reference_pool(tree, parent_id, config, topology_config)
    if not pool:
        return None, None
    used = set(tree.used_crossover_pairs)
    unused = [i for i in pool if (parent_id, i) not in used]
    code = tree.nodes[parent_id].code
    distance = {i: code_distance(code, tree.nodes[i].code) for i in (unused or pool)}
    best = max(distance, key=lambda i: (distance[i], -float(tree.nodes[i].fitness), -i))  # type: ignore[arg-type]
    return best, distance[best]


def select_explore_references(
    tree: TreePopulation,
    parent_id: int,
    config: AllocationConfig = AllocationConfig(),
    topology_config: TopologyConfig = TopologyConfig(),
) -> tuple[int, ...]:
    """From the representatives other than the parent, take the best one, then repeatedly add the one
    whose smallest AST distance to the parent and the already chosen ones is largest.

    No quality floor applies: for Explore, a weak approach and its outcome are also informative.
    """
    pool = [i for i in representatives(tree, topology_config) if i != parent_id]
    chosen = sorted(pool, key=lambda i: (tree.nodes[i].fitness, i))[:min(1, config.explore_references)]  # type: ignore[arg-type]
    while len(chosen) < min(config.explore_references, len(pool)):
        chosen.append(max(
            (i for i in pool if i not in chosen),
            key=lambda i: (min(code_distance(tree.nodes[j].code, tree.nodes[i].code) for j in (parent_id, *chosen)),
                           -float(tree.nodes[i].fitness), -i),  # type: ignore[arg-type]
        ))
    return tuple(chosen)


def allocate_step(
    tree: TreePopulation,
    rng: random.Random,
    config: AllocationConfig = AllocationConfig(),
    topology_config: TopologyConfig = TopologyConfig(),
) -> AllocationDecision:
    """Select one representative parent, one evolution operator, and its crossover reference or Explore references."""
    parent_id, probability = select_parent(tree, rng, config, topology_config)
    operators: list[Operator] = ["Refine", "Explore", "Crossover"]
    operator = rng.choices(operators, weights=[config.refine_weight, config.explore_weight,
                                             config.crossover_weight], k=1)[0]
    ref_id, ref_dist, explore_ids = None, None, ()
    if operator == "Crossover":
        ref_id, ref_dist = select_crossover_reference(tree, parent_id, config, topology_config)
        if ref_id is None:
            operator = "Refine"
    elif operator == "Explore":
        explore_ids = select_explore_references(tree, parent_id, config, topology_config)
    return AllocationDecision(parent_id, operator, ref_id, probability, ref_dist, explore_ids)
