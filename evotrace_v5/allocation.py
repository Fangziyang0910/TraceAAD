"""分叉局部非平稳后验预算分配，以及基于树谱系距离的供体与分支代表选取。"""

import math
from collections import defaultdict
from dataclasses import dataclass, field

from .models import ROOT_ID
from .topology import better

RECENCY_DECAY = 0.85


@dataclass
class History:
    start: float
    receipts: list[int] = field(default_factory=list)
    best: float = field(init=False)
    step_gains: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.best = self.start

    @property
    def count(self) -> int:
        return len(self.receipts)

    def observe(self, evaluation_id: int, score: float | None) -> None:
        self.receipts.append(evaluation_id)
        if score is not None and better(score, self.best):
            self.step_gains.append(self.best - score)
            self.best = score
        else:
            self.step_gains.append(0.0)


class Ledger:
    def __init__(self):
        self.direct: dict[int, History] = {}
        self.branch: dict[int, History] = {}

    def add(self, program) -> None:
        self.direct[program.id] = History(program.evaluation.fitness)
        self.branch[program.id] = History(program.evaluation.fitness)

    def reduce(self, event) -> None:
        if event.kind == "path":
            self.branch[event.target] = self.branch[event.removed[0]]
        for node in event.removed:
            del self.direct[node], self.branch[node]

    def record_attempt(self, tree, archive, attempt) -> None:
        parent = attempt.parent_id
        for row in attempt.evaluations:
            p = archive[row["program_id"]]
            score = None
            if p.valid:
                if p.duplicate_of is None and p.id not in self.direct:
                    self.add(p)
                score = p.evaluation.fitness
            if parent == ROOT_ID:
                continue
            self.direct[parent].observe(row["id"], score)
            node = parent
            while node != ROOT_ID:
                self.branch[node].observe(row["id"], score)
                node = tree.parents[node]


def choose_direction(options, rng, *, decay: float = RECENCY_DECAY):
    """在单个分叉比较候选方向：(destination, history)，直接改写记为 destination=None。"""
    frontier = min(history.best for _, history in options)
    records = []
    for target, history in options:
        n = history.count
        gap = history.best - frontier
        weighted_s = weighted_total = 0.0
        for idx, g in enumerate(history.step_gains):
            if g > 0.0 and better(history.best - g, frontier):
                weight = decay ** (n - 1 - idx)
                weighted_s += weight
                weighted_total += weight * (g - gap)
        records.append(
            {
                "target": target,
                "best": history.best,
                "gap": gap,
                "n": n,
                "s": weighted_s,
                "total": weighted_total,
            }
        )
    positives = math.fsum(r["s"] for r in records)
    scale = math.fsum(r["total"] for r in records) / positives if positives > 0.0 else None
    for row in records:
        s, n = row["s"], row["n"]
        chance = (s + 1.0) / (n + 2.0)
        amplitude = None if scale is None else (scale + row["total"]) / (s + 1.0)
        draw = rng.betavariate(s + 1.0, max(0.0, n - s) + 1.0)
        row.update(
            probability=chance,
            amplitude=amplitude,
            estimate=None if amplitude is None else chance * amplitude,
            draw=draw,
            value=draw if amplitude is None else draw * amplitude,
        )
    high = max(row["value"] for row in records)
    winners = [row["target"] for row in records if row["value"] == high]
    return (winners[0] if len(winners) == 1 else rng.choice(winners)), records


def choose_parent(tree, ledger, rng):
    """自虚拟根沿现役树逐层路由，选出本轮待改写的主父代。"""
    children = defaultdict(list)
    for child, parent in sorted(tree.parents.items()):
        children[parent].append(child)
    parent = ROOT_ID
    trace = []
    while True:
        options = ([] if parent == ROOT_ID else [(None, ledger.direct[parent])]) + [
            (child, ledger.branch[child]) for child in children[parent]
        ]
        if not options:
            raise ValueError("The tree has no active program")
        if len(options) == 1:
            target = options[0][0]
        else:
            target, records = choose_direction(options, rng)
            trace.append({"fork": parent, "options": records, "chosen": target})
        if target is None:
            return parent, trace
        parent = target


def _lca_and_dist(pa: tuple[int, ...], pb: tuple[int, ...]) -> tuple[int, int]:
    lca = 0
    for x, y in zip(pa, pb):
        if x != y:
            break
        lca += 1
    return lca, (len(pa) - lca) + (len(pb) - lca)


def tree_distance(tree, left: int, right: int) -> int:
    """两节点在现役树（经虚拟根相连）上的最短边数距离。"""
    return _lca_and_dist(tree.path_to(left), tree.path_to(right))[1]


def _select_donor(tree, others: list[int], parent: int, rng) -> int:
    """优先从非直系祖先/后代的跨分支节点中，按树上谱系距离加权抽取供体。"""
    p_path = tree.path_to(parent)
    cross, fallback = [], []
    for i in others:
        q_path = tree.path_to(i)
        lca, dist = _lca_and_dist(p_path, q_path)
        (fallback if lca == min(len(p_path), len(q_path)) else cross).append((i, dist))
    nodes, weights = zip(*(cross or fallback))
    return rng.choices(nodes, weights=weights, k=1)[0]


def _select_explore_references(tree, archive, others: list[int], parent: int, limit: int = 5) -> list[int]:
    """优先覆盖其他顶级分支的最优代表，不足时按最大最小树距离（同距取优分）补齐。"""
    if not others:
        return []
    paths = {i: tree.path_to(i) for i in (*others, parent)}
    parent_root = paths[parent][0] if paths[parent] else None

    by_root: dict[int, list[int]] = defaultdict(list)
    for i in others:
        root_child = paths[i][0] if paths[i] else i
        if root_child != parent_root:
            by_root[root_child].append(i)

    champions = sorted(
        (min(group, key=lambda i: (archive[i].fitness, i)) for group in by_root.values()),
        key=lambda i: (archive[i].fitness, i),
    )
    selected = champions[:limit]
    remaining = [i for i in others if i not in selected]
    while len(selected) < limit and remaining:
        anchors = (parent, *selected)
        best_next = min(
            remaining,
            key=lambda c, anchors=anchors: (
                -min(_lca_and_dist(paths[c], paths[a])[1] for a in anchors),
                archive[c].fitness,
                c,
            ),
        )
        selected.append(best_next)
        remaining.remove(best_next)
    return selected


def choose_operator(tree, archive, parent, rng):
    """按固定先验 (0.50, 0.30, 0.20) 抽取算子，并利用树谱系结构选取跨分支供体或分支代表。"""
    operator = rng.choices(("Refine", "Crossover", "Explore"), (0.50, 0.30, 0.20))[0]
    if operator == "Refine":
        return operator, None, []
    others = [i for i in sorted(tree.parents) if i != parent and archive[i].valid]
    if operator == "Crossover":
        return (operator, _select_donor(tree, others, parent, rng), []) if others else ("Refine", None, [])
    return operator, None, _select_explore_references(tree, archive, others, parent, limit=5)
