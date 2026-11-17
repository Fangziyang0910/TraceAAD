"""预算分配与参考选择：ShinkaEvolve 加权父代采样，配合质量池内最大 AST 距离交叉参考。"""

from dataclasses import dataclass
import math
import random
from statistics import median

from .models import Operator, TreePopulation
from .topology import code_distance, identity

OPERATORS: tuple[Operator, ...] = ("Refine", "Explore", "Crossover")
OPERATOR_WEIGHTS = (0.45, 0.30, 0.25)
REFERENCE_QUALITY_FLOOR = 0.30


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0.0 else (z := math.exp(x)) / (1.0 + z)


@dataclass(frozen=True)
class AllocationConfig:
    parent_selection_lambda: float = 10.0
    full_pool_threshold: int = 12

    def __post_init__(self) -> None:
        if not math.isfinite(self.parent_selection_lambda) or self.parent_selection_lambda < 0:
            raise ValueError("parent_selection_lambda must be finite and nonnegative")
        if self.full_pool_threshold < 0:
            raise ValueError("full_pool_threshold must be nonnegative")


def parent_probabilities(
    tree: TreePopulation, config: AllocationConfig = AllocationConfig()
) -> dict[int, float]:
    """按 ShinkaEvolve 中位数/MAD 归一化分差与累计子代数惩罚计算父代抽样概率。"""
    feasible = sorted(nid for nid, node in tree.nodes.items() if node.fitness is not None)
    if not feasible:
        return {}
    scores = sorted(float(tree.nodes[nid].fitness) for nid in feasible)  # type: ignore[arg-type]
    med = float(median(scores))
    mad = float(median([abs(s - med) for s in scores]))
    if mad <= 1e-9:
        nonzero = [abs(s - med) for s in scores if abs(s - med) > 1e-9]
        scale = max(float(median(nonzero)) if nonzero else 0.0, 1e-6)
    else:
        scale = max(mad, 1e-6)

    weights = {
        nid: _sigmoid(config.parent_selection_lambda * (med - float(tree.nodes[nid].fitness)) / scale)  # type: ignore[arg-type]
        / (1.0 + max(0, tree.nodes[nid].children_count))
        for nid in feasible
    }
    total = sum(weights.values())
    if total <= 0.0:
        return {nid: 1.0 / len(feasible) for nid in feasible}
    return {nid: w / total for nid, w in weights.items()}


def eligible_reference_pool(
    tree: TreePopulation, parent_id: int, config: AllocationConfig = AllocationConfig()
) -> list[int]:
    if parent_id not in tree.nodes or tree.nodes[parent_id].fitness is None:
        raise ValueError("Crossover parent must be a retained feasible node")
    ordered = sorted(
        (nid for nid, node in tree.nodes.items() if node.fitness is not None),
        key=lambda nid: (tree.nodes[nid].fitness, nid),  # type: ignore[arg-type]
    )
    if len(ordered) <= config.full_pool_threshold:
        pool = ordered
    else:
        denom = max(1, len(ordered) - 1)
        pool = [nid for rank, nid in enumerate(ordered) if (1.0 - rank / denom) >= REFERENCE_QUALITY_FLOOR]
    return [nid for nid in pool if nid != parent_id]


def select_crossover_reference(
    tree: TreePopulation, parent_id: int, config: AllocationConfig = AllocationConfig()
) -> int | None:
    """在合格参考池中选择标准化源码不同且 AST 结构距离最大的参考节点，优先未配对组合。"""
    parent_code = tree.nodes[parent_id].code
    parent_ident = identity(parent_code)
    distinct = [cid for cid in eligible_reference_pool(tree, parent_id, config)
                if identity(tree.nodes[cid].code) != parent_ident]
    if not distinct:
        return None
    used = set(tree.used_crossover_pairs)
    candidates = [cid for cid in distinct if (parent_id, cid) not in used] or distinct
    return max(
        candidates,
        key=lambda cid: (code_distance(parent_code, tree.nodes[cid].code), -float(tree.nodes[cid].fitness), -cid),  # type: ignore[arg-type]
    )


def allocate_step(
    tree: TreePopulation, rng: random.Random, config: AllocationConfig = AllocationConfig()
) -> tuple[int, Operator, int | None]:
    probs = parent_probabilities(tree, config)
    if not probs:
        raise ValueError("TreePopulation has no feasible algorithms to select from")
    parent_id = rng.choices(list(probs), weights=list(probs.values()), k=1)[0]
    operator = rng.choices(OPERATORS, weights=OPERATOR_WEIGHTS, k=1)[0]
    if operator == "Crossover":
        ref_id = select_crossover_reference(tree, parent_id, config)
        return (parent_id, "Refine", None) if ref_id is None else (parent_id, "Crossover", ref_id)
    return parent_id, operator, None
