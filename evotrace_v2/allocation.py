"""预算分配与参考选择：路线最大权重与路线内相对权重采样，配合质量池最大 AST 距离交叉参考。"""

from collections import defaultdict
from dataclasses import dataclass
import math
import random
from statistics import median

from .models import Operator, TreePopulation
from .topology import TopologyConfig, code_distance, identity, quality_representatives

OPERATORS: tuple[Operator, ...] = ("Refine", "Explore", "Crossover")
OPERATOR_WEIGHTS = (0.45, 0.30, 0.25)
REFERENCE_QUALITY_FLOOR = 0.30


@dataclass(frozen=True)
class AllocationConfig:
    parent_selection_lambda: float = 10.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.parent_selection_lambda) or self.parent_selection_lambda < 0:
            raise ValueError("parent_selection_lambda must be finite and nonnegative")


def parent_probabilities(
    tree: TreePopulation,
    config: AllocationConfig = AllocationConfig(),
    topology_config: TopologyConfig = TopologyConfig(),
) -> dict[int, float]:
    """按路线最大节点权重在路线间分配，路线内部按相对权重分配。"""
    feasible = sorted(nid for nid, node in tree.nodes.items() if node.fitness is not None)
    if not feasible:
        return {}

    reps = quality_representatives(tree, topology_config)
    scores = sorted(float(tree.nodes[nid].fitness) for nid in reps)  # type: ignore[arg-type]
    med = float(median(scores))
    mad = float(median([abs(s - med) for s in scores]))
    if mad <= 1e-9:
        nonzero = [abs(s - med) for s in scores if abs(s - med) > 1e-9]
        scale = max(float(median(nonzero)) if nonzero else 0.0, 1e-6)
    else:
        scale = max(mad, 1e-6)

    log_weights: dict[int, float] = {}
    routes: dict[int, list[int]] = defaultdict(list)
    for nid in feasible:
        node = tree.nodes[nid]
        x = config.parent_selection_lambda * (med - float(node.fitness)) / scale  # type: ignore[arg-type]
        log_sig = (-math.log1p(math.exp(-x)) if x < 40.0 else 0.0) if x >= 0.0 else (
            x - math.log1p(math.exp(x)) if x > -40.0 else x
        )
        log_weights[nid] = log_sig - math.log(1.0 + max(0, node.children_count))
        routes[tree.direction_of(nid)].append(nid)

    route_max = {did: max(log_weights[nid] for nid in members) for did, members in routes.items()}
    top_log = max(route_max.values())
    route_raw = {did: math.exp(lw - top_log) for did, lw in route_max.items()}
    total_route = sum(route_raw.values())
    if total_route <= 0.0:
        return {nid: 1.0 / len(feasible) for nid in feasible}

    probs: dict[int, float] = {}
    for did, members in routes.items():
        p_route = route_raw[did] / total_route
        m_log = route_max[did]
        inner_raw = {nid: math.exp(log_weights[nid] - m_log) for nid in members}
        inner_sum = sum(inner_raw.values())
        for nid in members:
            probs[nid] = p_route / len(members) if inner_sum <= 0.0 else max(
                math.ulp(0.0), p_route * (inner_raw[nid] / inner_sum)
            )
    return probs


def eligible_reference_pool(tree: TreePopulation, parent_id: int) -> list[int]:
    if parent_id not in tree.nodes or tree.nodes[parent_id].fitness is None:
        raise ValueError("Crossover parent must be a retained feasible node")
    own = identity(tree.nodes[parent_id].code)
    ordered = sorted(
        (nid for nid, node in tree.nodes.items()
         if node.fitness is not None and nid != parent_id and identity(node.code) != own),
        key=lambda nid: (tree.nodes[nid].fitness, nid),  # type: ignore[arg-type]
    )
    denom = max(1, len(ordered) - 1)
    return [nid for rank, nid in enumerate(ordered) if 1.0 - rank / denom >= REFERENCE_QUALITY_FLOOR]


def allocate_step(
    tree: TreePopulation,
    rng: random.Random,
    config: AllocationConfig = AllocationConfig(),
    topology_config: TopologyConfig = TopologyConfig(),
) -> tuple[int, Operator, int | None]:
    probs = parent_probabilities(tree, config, topology_config)
    if not probs:
        raise ValueError("TreePopulation has no feasible algorithms to select from")
    parent_id = rng.choices(list(probs), weights=list(probs.values()), k=1)[0]
    operator = rng.choices(OPERATORS, weights=OPERATOR_WEIGHTS, k=1)[0]
    if operator == "Crossover":
        pool = eligible_reference_pool(tree, parent_id)
        if not pool:
            return parent_id, "Refine", None
        used = set(tree.used_crossover_pairs)
        candidates = [cid for cid in pool if (parent_id, cid) not in used] or pool
        code = tree.nodes[parent_id].code
        ref_id = max(candidates, key=lambda cid: (code_distance(code, tree.nodes[cid].code), -float(tree.nodes[cid].fitness), -cid))  # type: ignore[arg-type]
        return parent_id, "Crossover", ref_id
    return parent_id, operator, None
