"""预算分配与参考选择：直接采用 ShinkaEvolve 加权父代采样，配合质量池内最不相似 AST 参考选择。"""

from collections import defaultdict
from dataclasses import dataclass
import math
import random
from statistics import median

from .models import Operator, TreePopulation
from .topology import TopologyConfig, code_distance, identity, quality_representatives


def _stable_sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass(frozen=True)
class AllocationConfig:
    """Shinka 加权父代采样、固定算子先验与交叉参考池配置。"""

    parent_selection_lambda: float = 10.0
    refine_weight: float = 0.45
    explore_weight: float = 0.30
    crossover_weight: float = 0.25
    reference_quality_floor: float = 0.30

    def __post_init__(self) -> None:
        if not math.isfinite(self.parent_selection_lambda) or self.parent_selection_lambda < 0:
            raise ValueError("parent_selection_lambda must be finite and nonnegative")
        weights = (self.refine_weight, self.explore_weight, self.crossover_weight)
        if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError("Operator weights must be nonnegative and sum to a positive value")
        if not 0.0 <= self.reference_quality_floor <= 1.0:
            raise ValueError("reference_quality_floor must be in [0, 1]")


@dataclass(frozen=True)
class AllocationDecision:
    """单步预算分配结果：主父代、生成算子、可选交叉参考及父代抽样概率。"""

    parent_id: int
    operator: Operator
    reference_id: int | None
    parent_probability: float
    reference_distance: float | None = None


def parent_probabilities(
    tree: TreePopulation,
    config: AllocationConfig = AllocationConfig(),
    topology_config: TopologyConfig = TopologyConfig(),
) -> dict[int, float]:
    """按路线最大节点权重与路线内同一权重分配可行节点的父代采样概率。

    最小化目标下，在采样前的当前树上重算质量代表，以其适应度中位数 m 与稳健偏差 s 归一化分差：
    ``w_v = sigmoid(lambda * (m - f_v) / s) / (1 + children_count_v)``，
    路线按 ``max_{u in D} w_u`` 竞争，内部按 ``w_v / sum_{u in D} w_u`` 分配。
    """
    feasible = sorted(
        (nid for nid, node in tree.nodes.items() if node.fitness is not None),
        key=lambda nid: nid,
    )
    if not feasible:
        return {}

    representatives = quality_representatives(tree, topology_config)
    scores = sorted(float(tree.nodes[nid].fitness) for nid in representatives)  # type: ignore[arg-type]
    med = float(median(scores))
    mad = float(median([abs(s - med) for s in scores]))
    if mad <= 1e-9:
        nonzero_devs = [abs(s - med) for s in scores if abs(s - med) > 1e-9]
        scale = max(float(median(nonzero_devs)) if nonzero_devs else 0.0, 1e-6)
    else:
        scale = max(mad, 1e-6)

    log_weights: dict[int, float] = {}
    routes: dict[int, list[int]] = defaultdict(list)
    for nid in feasible:
        node = tree.nodes[nid]
        x = config.parent_selection_lambda * (med - float(node.fitness)) / scale  # type: ignore[arg-type]
        if x >= 0.0:
            log_sig = -math.log1p(math.exp(-x)) if x < 40.0 else 0.0
        else:
            log_sig = x - math.log1p(math.exp(x)) if x > -40.0 else x
        log_weights[nid] = log_sig - math.log(1.0 + max(0, node.children_count))
        routes[tree.direction_of(nid)].append(nid)

    route_max_logs = {did: max(log_weights[nid] for nid in members) for did, members in routes.items()}
    top_route_log = max(route_max_logs.values())
    route_raw = {did: math.exp(lw - top_route_log) for did, lw in route_max_logs.items()}
    total_route = sum(route_raw.values())
    if total_route <= 0.0:
        uniform = 1.0 / len(feasible)
        return {nid: uniform for nid in feasible}

    probs: dict[int, float] = {}
    for did, members in routes.items():
        p_route = route_raw[did] / total_route
        m_log = route_max_logs[did]
        inner_raw = {nid: math.exp(log_weights[nid] - m_log) for nid in members}
        inner_sum = sum(inner_raw.values())
        if inner_sum <= 0.0:
            share = p_route / len(members)
            for nid in members:
                probs[nid] = share
        else:
            for nid in members:
                # Avoid a numerical zero for an eligible node when the quality
                # scale collapses. This is machine precision, not a budget floor.
                probs[nid] = max(math.ulp(0.0), p_route * (inner_raw[nid] / inner_sum))
    return probs


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
    denom = max(1, len(ordered) - 1)
    return [nid for rank, nid in enumerate(ordered)
            if 1.0 - rank / denom >= config.reference_quality_floor]


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
    topology_config: TopologyConfig = TopologyConfig(),
) -> AllocationDecision:
    """执行一步预算分配：抽样主父代与生成算子，若为 Crossover 则匹配最不相似参考（无可用异构参考时回退为 Refine）。"""
    probs = parent_probabilities(tree, config, topology_config)
    if not probs:
        raise ValueError("TreePopulation has no feasible algorithms to select from")

    ids = list(probs.keys())
    weights = [probs[nid] for nid in ids]
    parent_id = rng.choices(ids, weights=weights, k=1)[0]

    operators: list[Operator] = ["Refine", "Explore", "Crossover"]
    op_weights = [config.refine_weight, config.explore_weight, config.crossover_weight]
    operator: Operator = rng.choices(operators, weights=op_weights, k=1)[0]

    ref_id, ref_dist = None, None
    if operator == "Crossover":
        ref_id, ref_dist = select_crossover_reference(tree, parent_id, config)
        if ref_id is None:
            operator = "Refine"

    return AllocationDecision(
        parent_id=parent_id,
        operator=operator,
        reference_id=ref_id,
        parent_probability=probs[parent_id],
        reference_distance=ref_dist,
    )
