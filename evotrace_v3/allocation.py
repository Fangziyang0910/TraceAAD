"""Parent selection combining global rank quality with lineage-shared evaluation effort."""

from dataclasses import dataclass
import math
import random

from .models import Operator, TreePopulation
from .topology import code_distance, distinct_individuals, identity, quality_scores


@dataclass(frozen=True)
class AllocationConfig:
    rank_half_life: float = 10.0  # Rank positions over which quality weight halves.
    refine_weight: float = 0.45
    explore_weight: float = 0.30
    crossover_weight: float = 0.25
    reference_quality_floor: float = 0.30

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


def lineage_peaks(tree: TreePopulation) -> dict[int, int]:
    """Map each retained feasible node to the best feasible ancestor on its retained path from the root.

    Smaller fitness wins; ties prefer the earlier node ID. Every feasible node lies on its
    own path, so this ancestor always exists.
    """
    best_on_path: dict[int, int | None] = {tree.root_id: None}
    for node_id in tree.nodes:
        path: list[int] = []
        cur = node_id
        while cur not in best_on_path:
            path.append(cur)
            cur = tree.edges[cur].parent_id
        peak = best_on_path[cur]
        for uid in reversed(path):
            fit = tree.nodes[uid].fitness
            if fit is not None and (
                peak is None or (fit, uid) < (tree.nodes[peak].fitness, peak)  # type: ignore[operator]
            ):
                peak = uid
            best_on_path[uid] = peak
    return {
        nid: pid
        for nid, pid in best_on_path.items()
        if nid != tree.root_id and pid is not None and tree.nodes[nid].fitness is not None
    }


def parent_probabilities(tree: TreePopulation,
                         config: AllocationConfig = AllocationConfig()) -> dict[int, float]:
    """Normalize 2**(-rank_gap / rank_half_life) / (1 + subtree_evaluations[a(i)]).

    Retained feasible individuals compete globally by measured rank quality while
    sharing cumulative evaluation effort with the best feasible algorithm ``a(i)``
    on their retained path from the root:
    - Descendants that have not improved upon ``a(i)`` share its ``subtree_evaluations``,
      keeping worse offspring below their parent and bounding the total weight of any
      non-improving family within a tree state by ``a(i)``'s base quality weight.
    - A descendant that strictly improves upon its path record becomes its own peak with
      fresh effort, while heavily explored lineages yield budget to less-explored branches.
    """
    quality = quality_scores(tree, distinct_individuals(tree))
    if not quality:
        return {}
    peaks = lineage_peaks(tree)
    best_quality = max(quality.values())
    log_weights = {
        i: -math.log(2) * (best_quality - quality[i]) * (len(quality) - 1)
           / config.rank_half_life - math.log1p(tree.nodes[peaks[i]].subtree_evaluations)
        for i in sorted(quality)
    }
    largest = max(log_weights.values())
    weights = {i: math.exp(value - largest) for i, value in log_weights.items()}
    total = math.fsum(weights.values())
    # Keep numerical underflow from becoming an eligibility rule at extreme settings.
    return {i: max(math.ulp(0.0), weight / total) for i, weight in weights.items()}


def select_parent(tree: TreePopulation, rng: random.Random,
                  config: AllocationConfig = AllocationConfig()) -> tuple[int, float]:
    probabilities = parent_probabilities(tree, config)
    if not probabilities:
        raise ValueError("TreePopulation has no feasible algorithms to select from")
    parent = rng.choices(list(probabilities), weights=list(probabilities.values()), k=1)[0]
    return parent, probabilities[parent]


def eligible_reference_pool(
    tree: TreePopulation,
    parent_id: int,
    config: AllocationConfig = AllocationConfig(),
) -> list[int]:
    """Return feasible crossover reference candidates ranked above ``reference_quality_floor``."""
    if parent_id not in tree.nodes or tree.nodes[parent_id].fitness is None:
        raise ValueError("Crossover parent must be a retained feasible node")

    own = identity(tree.nodes[parent_id].code)
    ordered = sorted(
        (nid for nid in distinct_individuals(tree)
         if nid != parent_id and identity(tree.nodes[nid].code) != own),
        key=lambda nid: (tree.nodes[nid].fitness, nid),  # type: ignore[arg-type]
    )
    quality = quality_scores(tree, ordered)
    return [nid for nid in ordered if quality[nid] >= config.reference_quality_floor]


def select_crossover_reference(
    tree: TreePopulation,
    parent_id: int,
    config: AllocationConfig = AllocationConfig(),
) -> tuple[int | None, float | None]:
    """Select the eligible reference with maximum AST structural distance from ``parent_id``."""
    pool = eligible_reference_pool(tree, parent_id, config)
    if not pool:
        return None, None

    parent_code = tree.nodes[parent_id].code
    parent_ident = identity(parent_code)
    used = set(tree.used_crossover_pairs)

    scored: list[tuple[float, int, bool]] = []
    for cand_id in pool:
        cand_code = tree.nodes[cand_id].code
        if identity(cand_code) == parent_ident:
            continue
        dist = code_distance(parent_code, cand_code)
        is_unused = (parent_id, cand_id) not in used
        scored.append((dist, cand_id, is_unused))

    if not scored:
        return None, None

    unused_candidates = [(dist, cid) for dist, cid, is_unused in scored if is_unused]
    active = unused_candidates if unused_candidates else [(dist, cid) for dist, cid, _ in scored]
    best_dist, best_id = max(
        active,
        key=lambda item: (item[0], -float(tree.nodes[item[1]].fitness), -item[1]),  # type: ignore[arg-type]
    )
    return best_id, best_dist


def allocate_step(
    tree: TreePopulation,
    rng: random.Random,
    config: AllocationConfig = AllocationConfig(),
) -> AllocationDecision:
    """Select one parent individual, one evolution operator, and optional crossover reference."""
    parent_id, probability = select_parent(tree, rng, config)
    operators: list[Operator] = ["Refine", "Explore", "Crossover"]
    operator = rng.choices(operators, weights=[config.refine_weight, config.explore_weight,
                                             config.crossover_weight], k=1)[0]
    ref_id, ref_dist = None, None
    if operator == "Crossover":
        ref_id, ref_dist = select_crossover_reference(tree, parent_id, config)
        if ref_id is None:
            operator = "Refine"
    return AllocationDecision(parent_id, operator, ref_id, probability, ref_dist)
