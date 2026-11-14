"""Global individual sampling using quality and historical subtree effort."""

from dataclasses import dataclass
import math
import random

from .models import Operator, TreePopulation
from .topology import code_distance, identity, quality_scores


@dataclass(frozen=True)
class AllocationConfig:
    quality_ratio: float = 10.0  # Best/worst quality weight ratio at equal effort.
    refine_weight: float = 0.45
    explore_weight: float = 0.30
    crossover_weight: float = 0.25
    reference_quality_floor: float = 0.30

    def __post_init__(self) -> None:
        if not math.isfinite(self.quality_ratio) or self.quality_ratio < 1:
            raise ValueError("quality_ratio must be finite and at least 1")
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


def parent_probabilities(tree: TreePopulation,
                         config: AllocationConfig = AllocationConfig()) -> dict[int, float]:
    """Normalize quality_ratio**rank / (1 + historical subtree evaluations).

    Every retained feasible individual competes directly. An ancestor's effort
    discounts only that ancestor; it is never an entry gate for its descendants.
    Counts express a budget preference, not evidence of exhausted potential.
    """
    quality = quality_scores(tree)
    if not quality:
        return {}
    log_weights = {i: math.log(config.quality_ratio) * quality[i]
                   - math.log1p(tree.nodes[i].subtree_evaluations) for i in sorted(quality)}
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
    """返回候选交叉参考编号列表（按适应度从优到劣排序）。

    所有规模共用相同质量门槛，排除主父代及相同源码。
    在实际可用的不同实现中排名，避免最优主父代占用参考资格。
    """
    if parent_id not in tree.nodes or tree.nodes[parent_id].fitness is None:
        raise ValueError("Crossover parent must be a retained feasible node")

    own = identity(tree.nodes[parent_id].code)
    ordered = sorted(
        (nid for nid, node in tree.nodes.items()
         if node.fitness is not None and nid != parent_id and identity(node.code) != own),
        key=lambda nid: (tree.nodes[nid].fitness, nid),  # type: ignore[arg-type]
    )
    quality = quality_scores(tree, ordered)
    return [nid for nid in ordered if quality[nid] >= config.reference_quality_floor]


def select_crossover_reference(
    tree: TreePopulation,
    parent_id: int,
    config: AllocationConfig = AllocationConfig(),
) -> tuple[int | None, float | None]:
    """在合格参考池中选择与主父代 AST 结构最不相似（距离最大）的参考节点。

    资格判定使用标准化源码等价性 ``identity(code)``（避免忽略变量身份的 AST 指纹误拒不同计算，如 ``x/y`` 与 ``y/x``）；
    排序偏好使用 ``code_distance`` 最大。优先选取尚未与 ``parent_id`` 配对过的候选；全部已配对时允许复用。
    """
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
    # 距离最大优先；距离相同时适应度更优、编号更早者优先
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
    """Select one individual, then one generation operator and optional reference."""
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
