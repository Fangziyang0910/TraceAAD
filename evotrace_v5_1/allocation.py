"""强聚焦竞争与三算子选材：中位数/MAD 陡峭 Sigmoid 质量权重 × 直接投入折扣。"""

from __future__ import annotations

import math
import random
from statistics import median

from .models import ROOT_ID, Operator, Program, TreePopulation

OPERATORS: tuple[Operator, ...] = ("Refine", "Explore", "Crossover")
OPERATOR_WEIGHTS: tuple[float, ...] = (0.45, 0.30, 0.25)


def _log_sigmoid(x: float) -> float:
    if x >= 0.0:
        return 0.0 if x >= 40.0 else -math.log1p(math.exp(-x))
    return x if x <= -40.0 else x - math.log1p(math.exp(x))


def _softmax_from_logs(log_weights: dict[int, float]) -> dict[int, float]:
    if not log_weights:
        return {}
    top = max(log_weights.values())
    raw = {nid: math.exp(lw - top) for nid, lw in log_weights.items()}
    total = sum(raw.values())
    if total <= 0.0 or not math.isfinite(total):
        uniform = 1.0 / len(log_weights)
        return {nid: uniform for nid in log_weights}
    return {nid: val / total for nid, val in raw.items()}


def log_quality_weights(
    tree: TreePopulation,
    archive: dict[int, Program],
    *,
    selection_lambda: float = 10.0,
) -> tuple[dict[int, float], float, float]:
    """计算全存活树的中位数 m、MAD 尺度 s 及对数质量因子 log q(u) = log σ(λ(m-f(u))/s)。"""
    active = sorted(tree.parents)
    if not active:
        return {}, 0.0, 1e-6
    scores = sorted(float(archive[nid].fitness) for nid in active)
    med = float(median(scores))
    mad = float(median([abs(x - med) for x in scores]))
    if mad <= 1e-9:
        nonzero = [abs(x - med) for x in scores if abs(x - med) > 1e-9]
        scale = max(float(median(nonzero)) if nonzero else 0.0, 1e-6)
    else:
        scale = max(mad, 1e-6)
    log_q = {
        nid: _log_sigmoid(selection_lambda * (med - float(archive[nid].fitness)) / scale)
        for nid in active
    }
    return log_q, med, scale


def parent_probabilities(
    tree: TreePopulation,
    archive: dict[int, Program],
    *,
    selection_lambda: float = 10.0,
) -> tuple[dict[int, float], dict[int, float], dict]:
    """计算主父代抽样概率 P(u) ∝ q(u) / (1 + n_u)，同时返回对数质量因子与诊断统计。"""
    log_q, med, scale = log_quality_weights(tree, archive, selection_lambda=selection_lambda)
    if not log_q:
        return {}, {}, {}
    log_w = {
        nid: lq - math.log(1.0 + max(0, tree.direct_evals.get(nid, 0)))
        for nid, lq in log_q.items()
    }
    probs = _softmax_from_logs(log_w)
    lineages = tree.active_lineages()
    meta = {
        "population_size": len(log_q),
        "active_lineages": len(lineages),
        "median_fitness": med,
        "mad_scale": scale,
    }
    return probs, log_q, meta


def explore_leaders(
    tree: TreePopulation, archive: dict[int, Program], *, limit: int = 5
) -> tuple[int, ...]:
    """选取至多 limit 条活跃顶级谱系的各自最优代表，按成绩由优到劣排列（不以同谱系补齐）。"""
    leaders: list[int] = []
    for members in tree.active_lineages().values():
        best_in_lineage = min(members, key=lambda nid: (float(archive[nid].fitness), nid))
        leaders.append(best_in_lineage)
    leaders.sort(key=lambda nid: (float(archive[nid].fitness), nid))
    return tuple(leaders[:limit])


def choose_donor(
    tree: TreePopulation,
    log_q: dict[int, float],
    parent_id: int,
    rng: random.Random,
) -> tuple[int | None, bool]:
    """从其他顶级谱系按对数质量因子 q(q) 抽样供体；单谱系回退至 V\\{p}。返回 (donor_id, cross_lineage)。"""
    parent_lineage = tree.lineage_root(parent_id)
    cross_pool = [nid for nid in sorted(tree.parents) if tree.lineage_root(nid) != parent_lineage]
    if cross_pool:
        pool, is_cross = cross_pool, True
    else:
        pool, is_cross = [nid for nid in sorted(tree.parents) if nid != parent_id], False
    if not pool:
        return None, False
    donor_probs = _softmax_from_logs({nid: log_q[nid] for nid in pool})
    chosen = rng.choices(pool, weights=[donor_probs[nid] for nid in pool], k=1)[0]
    return chosen, is_cross


def choose_step(
    tree: TreePopulation,
    archive: dict[int, Program],
    rng: random.Random,
    *,
    selection_lambda: float = 10.0,
    explore_references: int = 5,
) -> tuple[int, Operator, int | None, tuple[int, ...], dict]:
    """按固定先验抽取算子，并完成主父代、跨谱系供体或探索代表选择。"""
    if not tree.parents:
        raise ValueError("TreePopulation has no active programs")

    operator: Operator = rng.choices(OPERATORS, weights=OPERATOR_WEIGHTS, k=1)[0]
    if operator == "Explore":
        leaders = explore_leaders(tree, archive, limit=explore_references)
        lineages = tree.active_lineages()
        trace = {
            "operator": "Explore",
            "population_size": len(tree.parents),
            "active_lineages": len(lineages),
            "explore_leaders": list(leaders),
        }
        return ROOT_ID, "Explore", None, leaders, trace

    probs, log_q, meta = parent_probabilities(tree, archive, selection_lambda=selection_lambda)
    active = sorted(probs)
    parent_id = rng.choices(active, weights=[probs[nid] for nid in active], k=1)[0]

    best_id = min(active, key=lambda nid: (float(archive[nid].fitness), nid))
    best_fit = float(archive[best_id].fitness)
    parent_fit = float(archive[parent_id].fitness)
    rel_gap = (parent_fit - best_fit) / max(abs(best_fit), 1e-12)

    reference_id: int | None = None
    cross_lineage: bool | None = None
    if operator == "Crossover":
        reference_id, cross_lineage = choose_donor(tree, log_q, parent_id, rng)
        if reference_id is None:
            operator = "Refine"
            cross_lineage = None

    trace = {
        **meta,
        "operator": operator,
        "parent_id": parent_id,
        "parent_prob": probs[parent_id],
        "parent_direct_evals": tree.direct_evals.get(parent_id, 0),
        "parent_is_champion": parent_id == best_id,
        "parent_gap_to_best": parent_fit - best_fit,
        "parent_rel_gap_to_best": rel_gap,
        "parent_lineage": tree.lineage_root(parent_id),
        "reference_id": reference_id,
        "cross_lineage": cross_lineage,
    }
    return parent_id, operator, reference_id, (), trace
